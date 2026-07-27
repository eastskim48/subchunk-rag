"""Shared runtime for compressors backed by a ColBERT candidate store."""

import os
import time

from chunk import RetrievableChunk
from colbert_artifact import ColBERTWindowArtifact
from compressor.base import Compressor
from compressor.token_budget import TokenBudgetMixin
from encoder.colbert import ColBERTEncoder, default_colbert_repo_path
from materialize.db_manifest import read_db_build_manifest


def _configured_db_dir() -> str | None:
    db_dir = os.getenv("DB_DIR")
    if db_dir:
        return db_dir
    dataset_path = os.getenv("DATASET_PATH")
    data_subdir = os.getenv("DATA_SUBDIR")
    if dataset_path and data_subdir:
        return os.path.join(dataset_path, data_subdir, "db")
    return None


def _resolve_configured_retrieval_chunk_size() -> int:
    """Read the retrieval-unit size recorded when the vector DB was built."""

    db_dir = _configured_db_dir()
    if db_dir is None:
        raise ValueError(
            "DB_DIR or DATASET_PATH/DATA_SUBDIR is required for retrieval chunk "
            "validation"
        )
    manifest = read_db_build_manifest(db_dir)
    if manifest is None:
        raise ValueError(f"DB build manifest is required for ColBERT runtime: {db_dir}")
    manifest_size = manifest.get("retrievable_chunk_size")
    if isinstance(manifest_size, bool) or not isinstance(manifest_size, int):
        raise ValueError(
            "DB build manifest retrievable_chunk_size must be an integer, "
            f"got {manifest_size!r}"
        )
    if manifest_size <= 0:
        raise ValueError(
            "DB build manifest retrievable_chunk_size must be positive, "
            f"got {manifest_size}"
        )
    return manifest_size


class ColBERTWindowCompressorBase(TokenBudgetMixin, Compressor):
    """Load the contextualized artifact and query encoder used online."""

    def __init__(
        self,
        *,
        initialize_token_budget: bool = True,
    ):
        super().__init__()
        if initialize_token_budget:
            self._initialize_token_budget()
        self.last_profile: dict[str, float | int] = {}
        artifact_dir = os.getenv("COLBERT_WINDOW_DIR")
        if not artifact_dir:
            dataset_path = os.getenv("DATASET_PATH")
            data_subdir = os.getenv("DATA_SUBDIR", "sent")
            if not dataset_path:
                raise ValueError(
                    "COLBERT_WINDOW_DIR or DATASET_PATH must be set for "
                    "colbert_subchunk compression"
                )
            artifact_dir = os.path.join(dataset_path, data_subdir, "colbert_window")

        model_name = os.getenv("COLBERT_MODEL_NAME", "colbert-ir/colbertv2.0")
        batch_size = int(os.getenv("COLBERT_BATCH_SIZE", "32"))
        repo_path = os.getenv("COLBERT_REPO_PATH") or default_colbert_repo_path()

        print(f"ColBERT window compression enabled. Loading artifact: {artifact_dir}")
        self.artifact = ColBERTWindowArtifact(artifact_dir)
        if getattr(self.artifact, "db_manifest_reference", None) is not None:
            db_dir = _configured_db_dir()
            if db_dir is None:
                raise ValueError(
                    "DB_DIR or DATASET_PATH/DATA_SUBDIR is required to validate "
                    "the ColBERT artifact DB manifest"
                )
            self.artifact.validate_db_manifest(db_dir)
        if initialize_token_budget:
            print(
                "ColBERT final budget tokenizer enabled: "
                f"{self.budget_tokenizer_name!r}"
            )
        artifact_model_name = self.artifact.index.get(
            "checkpoint_name"
        ) or self.artifact.index.get("model_name")
        if artifact_model_name != model_name:
            raise ValueError(
                "COLBERT_MODEL_NAME does not match the ColBERT window artifact: "
                f"runtime={model_name!r}, artifact={artifact_model_name!r}"
            )
        artifact_query_maxlen = int(self.artifact.index["official_query_maxlen"])
        configured_query_maxlen = os.getenv("COLBERT_QUERY_MAXLEN")
        query_maxlen = (
            int(configured_query_maxlen)
            if configured_query_maxlen is not None
            else artifact_query_maxlen
        )
        if query_maxlen < 4:
            raise ValueError(
                f"COLBERT_QUERY_MAXLEN must be at least 4, got {query_maxlen}"
            )
        configured_query_minlen = os.getenv("COLBERT_QUERY_MINLEN")
        query_minlen = (
            int(configured_query_minlen)
            if configured_query_minlen not in {None, ""}
            else None
        )
        query_truncation_side = (
            os.getenv("COLBERT_QUERY_TRUNCATION_SIDE", "right").strip().lower()
        )
        self.encoder = ColBERTEncoder(
            model_name=model_name,
            repo_path=repo_path,
            device="cpu",
            batch_size=batch_size,
            query_maxlen=query_maxlen,
            query_minlen=query_minlen,
            query_truncation_side=query_truncation_side,
            disable_cpu_extension=True,
            verify_tensorization=False,
        )
        self.query_encoder_warmup_time = 0.0

    def warmup_query_encoder(self) -> float:
        start = time.perf_counter()
        self.encoder.encode_queries(["warmup query"])
        self.query_encoder_warmup_time = time.perf_counter() - start
        return self.query_encoder_warmup_time

    def clear_inter_batch_cache(self) -> None:
        self.artifact.retrievable_vectors_cache.clear()

    @staticmethod
    def _empty_profile() -> dict[str, float | int]:
        return {
            "query_encode_time": 0.0,
            "budget_time": 0.0,
            "artifact_lookup_time": 0.0,
            "region_spec_time": 0.0,
            "region_object_time": 0.0,
            "sentence_maxsim_time": 0.0,
            "region_score_time": 0.0,
            "sort_time": 0.0,
            "select_time": 0.0,
            "build_output_time": 0.0,
            "query_count": 0,
            "retrieved_doc_count": 0,
            "region_count": 0,
            "unique_sentence_count": 0,
            "sentence_token_count": 0,
        }

    @staticmethod
    def _profile_add(profile: dict[str, float | int] | None, key: str, value):
        if profile is not None:
            profile[key] = profile.get(key, 0) + value

    @staticmethod
    def _build_unselected_document(doc: RetrievableChunk) -> RetrievableChunk:
        cloned = doc.clone()
        cloned.cacheables = []
        return cloned
