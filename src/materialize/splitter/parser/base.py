"""Abstract parser interface for source-aligned units."""

from abc import ABC, abstractmethod

from materialize.splitter.types import ParsedUnit


class UnitParser(ABC):
    """Parse document text into ordered source-aligned units."""

    @abstractmethod
    def parse(self, text: str, token_ids: list[int]) -> list[ParsedUnit]:
        raise NotImplementedError
