from typing import Dict, Any, Iterable, TypeVar, Type, Set, Optional
import numpy as np

from smqtk_image_io import ImageReader
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from smqtk_dataprovider import DataElement
from smqtk_descriptors.utils.parallel import parallel_map
from smqtk_descriptors.interfaces.descriptor_generator import \
    DescriptorGenerator
from smqtk_descriptors.interfaces.image_descriptor_generator import \
    ImageDescriptorGenerator

T = TypeVar("T", bound="ImageDescriptorGeneratorWrapper")


class ImageDescriptorGeneratorWrapper (DescriptorGenerator):
    """
    Wrapper around some image descriptor generator, mediating image reading
    into a numpy array.

    :param smqtk_image_io.ImageReader image_reader:
        ImageReader algorithm instance to facilitate reading image bytes
        and producing pixel matrices.
    :param smqtk_descriptors.interfaces.descriptor_generator_image.
    ImageDescriptorGenerator image_descriptor_generator:
        Image descriptor generator instance to perform descriptor generation
        on loaded image matrices.
    :param None|int image_load_threads:
        Optional integer number of threads to parallelize image loading.
        Thread parallelism only occurs if this is a positive integer
        greater than 1.
    """

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        c = super(ImageDescriptorGeneratorWrapper, cls).get_default_config()
        c['image_reader'] = make_default_config(ImageReader.get_impls())
        c['image_descriptor_generator'] = \
            make_default_config(ImageDescriptorGenerator.get_impls())
        return c

    @classmethod
    def from_config(
        cls: Type[T],
        config_dict: Dict,
        merge_default: bool = True
    ) -> T:
        config_dict = dict(config_dict)  # shallow copy for modifying
        config_dict['image_reader'] = from_config_dict(
            config_dict.get('image_reader', {}),
            ImageReader.get_impls()
        )
        config_dict['image_descriptor_generator'] = from_config_dict(
            config_dict.get('image_descriptor_generator', {}),
            ImageDescriptorGenerator.get_impls()
        )
        return super(ImageDescriptorGeneratorWrapper, cls).from_config(
            config_dict, merge_default=merge_default
        )

    def __init__(
        self,
        image_reader: ImageReader,
        image_descriptor_generator: ImageDescriptorGenerator,
        image_load_threads: Optional[int] = None
    ):
        super(ImageDescriptorGeneratorWrapper, self).__init__()
        self._image_reader = image_reader
        self._image_descr_generator = image_descriptor_generator
        self._image_load_threads = image_load_threads

    def get_config(self) -> Dict[str, Any]:
        return {
            "image_reader": to_config_dict(self._image_reader),
            "image_descriptor_generator":
                to_config_dict(self._image_descr_generator),
            "image_load_threads": self._image_load_threads,
        }

    def valid_content_types(self) -> Set[str]:
        return self._image_reader.valid_content_types()

    def is_valid_element(self, data_element: DataElement) -> bool:
        # Check element validity though the ImageReader algorithm instance.
        return self._image_reader.is_valid_element(data_element)

    def _generate_arrays(
        self,
        data_iter: Iterable[DataElement]
    ) -> Iterable[np.ndarray]:
        ir_load = self._image_reader.load_as_matrix
        i_load_threads = self._image_load_threads

        if i_load_threads and i_load_threads > 1:
            return self._image_descr_generator.generate_arrays_from_images(
                parallel_map(ir_load, data_iter,
                             cores=i_load_threads)
            )
        else:
            return self._image_descr_generator.generate_arrays_from_images(
                ir_load(d) for d in data_iter
            )
