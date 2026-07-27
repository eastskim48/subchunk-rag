"""Construct a parsed-unit grouper from its configuration name."""

from materialize.splitter.grouper.coref import CorefPronounDPGrouper
from materialize.splitter.grouper.pronoun import PronounDPGrouper


def build_grouper(name: str | None, tokenizer):
    if name is None:
        return None
    if name == "pronoun_dp":
        return PronounDPGrouper(tokenizer=tokenizer)
    if name == "coref_pronoun_dp":
        return CorefPronounDPGrouper(tokenizer=tokenizer)
    raise ValueError(f"unsupported grouper: {name}")
