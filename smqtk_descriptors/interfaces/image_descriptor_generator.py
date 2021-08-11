import abc
from typing import Iterable
import numpy as np

from smqtk_core import Configurable, Pluggable


class ImageDescriptorGenerator (Configurable, Pluggable):
    """
    Algorithm that generates feature descriptor arrays for input image matrices
    as ``numpy.ndarray`` type arrays.
    """

    @abc.abstractmethod
    def generate_arrays_from_images(
        self,
        img_mat_iter: Iterable[np.ndarray]
    ) -> Iterable[np.ndarray]:
        """
        Generate descriptor vector elements for input image matrices.

        :param img_mat_iter:
            Iterable of numpy arrays representing input image matrices.

        :raises RuntimeError: Descriptor extraction failure of some kind.

        :return: Iterable of numpy arrays in parallel association with the
            input image matrices.
        """
