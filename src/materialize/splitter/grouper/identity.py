"""Identity grouping for one cacheable candidate per parsed unit."""

from materialize.splitter.grouper.base import UnitGrouper
from materialize.splitter.types import ParsedUnit


class IdentityGrouper(UnitGrouper):
    """Keep every parsed unit as its own group."""

    name = "identity"

    def group(self, units: list[ParsedUnit]) -> list[list[int]]:
        return [[index] for index in range(len(units))]
