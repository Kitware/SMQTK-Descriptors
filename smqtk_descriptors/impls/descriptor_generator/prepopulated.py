import logging
import numpy as np
from typing import Any, Dict, Iterable, Set, TypeVar

from smqtk_dataprovider import DataElement
from smqtk_descriptors import DescriptorGenerator

LOG = logging.getLogger(__name__)

__all__ = ["PrePopulatedDescriptorGenerator"]
T = TypeVar("T", bound="PrePopulatedDescriptorGenerator")


class PrePopulatedDescriptorGenerator(DescriptorGenerator):
    """
    This class is to be used in the config when the descriptor set is already
    prepopulated.  This allows, for example, an IQR process where the
    descriptors are already known or have been previously generated using some
    external process. Calling the _generate_arrays() method will not work and
    will raise an AssertionError.
    """

    def valid_content_types(self) -> Set:
        return set()

    def _generate_arrays(
        self, data_iter: Iterable[DataElement]
    ) -> Iterable[np.ndarray]:
        raise AssertionError(
            "Method should not be called since descriptors are prepopulated."
        )

    def get_config(self) -> Dict[str, Any]:
        return {}
