"""DCLDE dataset builder stub."""

from anonymous_datasets.schema import DatasetInfo, DatasetSource, Version
from anonymous_datasets.utils import BaseDatasetBuilder


class DCLDE(BaseDatasetBuilder):
    VERSION = Version("0.0.0")
    SOURCE = DatasetSource(
        homepage="TBD",
        citation="TBD",
        assets={},
    )

    def _info(self) -> DatasetInfo:  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")

    def _split_generators(self):  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")

    def _generate_examples(self, **kwargs):  # pragma: no cover
        raise NotImplementedError("DCLDE builder not implemented yet.")
