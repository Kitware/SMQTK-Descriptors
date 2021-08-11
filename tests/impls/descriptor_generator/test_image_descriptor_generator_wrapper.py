from typing import Optional, Set, Dict, Iterable, Any
import unittest
from unittest import mock
import numpy as np


from smqtk_descriptors.impls.descriptor_generator.image_descriptor_generator_wrapper import (
    ImageDescriptorGeneratorWrapper,
    ImageDescriptorGenerator,
    ImageReader
)
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from smqtk_dataprovider import DataElement
from smqtk_core.configuration import configuration_test_helper
from smqtk_descriptors.utils.parallel import (
    parallel_map, ParallelResultsIterator
)


# Stub classes for testing
class StubImageReader (ImageReader):

    def _load_as_matrix(
        self,
        data_element: DataElement,
        pixel_crop: Optional[AxisAlignedBoundingBox] = None
    ) -> Optional[np.ndarray]:
        return iter([])

    def get_config(self) -> Dict:
        return {}

    def valid_content_types(self) -> Set:
        return set()


class StubImageDG (ImageDescriptorGenerator):

    def generate_arrays_from_images(
        self,
        img_mat_iter: Iterable[np.ndarray]
    ) -> Iterable[np.ndarray]:
        return iter([])

    def get_config(self) -> Dict[str, Any]:
        return {}


class TestImageDescriptorGeneratorWrapper (unittest.TestCase):
    """ Unit tests for ImageDescriptorGeneratorWrapper """

    def test_configuration(self) -> None:
        """ Test standard configuration capabilities """
        dummy_image_reader = StubImageReader()
        dummy_image_dg = StubImageDG()

        inst = ImageDescriptorGeneratorWrapper(dummy_image_reader,
                                               dummy_image_dg,
                                               image_load_threads=3)

        for inst_i in configuration_test_helper(inst):
            assert inst_i._image_load_threads == 3
            assert isinstance(inst_i._image_reader, StubImageReader)
            assert isinstance(inst_i._image_descr_generator, StubImageDG)

    def test_valid_content_types(self) -> None:
        """ This implementation should defer to the content types reported
        by the configured image reader."""
        expected_vct = {'the expected return set'}

        m_img_dg = mock.Mock(spec=ImageDescriptorGenerator)
        m_img_reader = mock.Mock(spec=ImageReader)
        m_img_reader.valid_content_types.return_value = expected_vct

        inst = ImageDescriptorGeneratorWrapper(m_img_reader, m_img_dg)
        assert inst.valid_content_types() == expected_vct
        m_img_reader.valid_content_types.assert_called_once()

    def test_is_valid_element(self) -> None:
        """ Test that ``is_valid_element`` defers to image reader impl due to
        special behavior of image reader interface."""
        m_img_dg = mock.Mock(spec=ImageDescriptorGenerator)

        expected_return = 'deferred result'
        m_img_reader = mock.Mock(spec=ImageReader)
        m_img_reader.is_valid_element.return_value = expected_return

        inst = ImageDescriptorGeneratorWrapper(m_img_reader, m_img_dg)

        m_ele = mock.Mock(spec=DataElement)
        assert inst.is_valid_element(m_ele) == expected_return
        m_img_reader.is_valid_element.assert_called_once_with(m_ele)

    def test_generate_arrays(self) -> None:
        """ Test that the underlying image descr generator is called with
        mock loaded matrices."""
        m_img_reader = mock.Mock(spec=ImageReader)
        # Mock image reader to return a matrix for an input something.
        m_img_reader.load_as_matrix.side_effect = lambda e: "matrix!"+e

        m_img_dg = mock.Mock(spec=ImageDescriptorGenerator)
        # Mock generator method to simply pass-through its input
        m_img_dg.generate_arrays_from_images.side_effect = \
            lambda it: ('descriptor!'+v for v in it)

        inst = ImageDescriptorGeneratorWrapper(m_img_reader, m_img_dg)

        # pass some dummy elements (strings) and observe that each are
        # "converted" to a matrix, and that the return iterator results
        # noinspection DuplicatedCode,PyTypeChecker
        ret = list(inst._generate_arrays(['a', 'b']))  # type: ignore

        # Image reader should have been called at least once per input.
        assert m_img_reader.load_as_matrix.call_count == 2
        m_img_reader.load_as_matrix.assert_any_call('a')
        m_img_reader.load_as_matrix.assert_any_call('b')

        # Image descriptor should have been called once with an iterator of the
        # combined "matrix" outputs.
        m_img_dg.generate_arrays_from_images.assert_called_once()
        assert ret == ['descriptor!matrix!a', 'descriptor!matrix!b']

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.image_descriptor_generator_wrapper.'
                'parallel_map', wraps=parallel_map)
    def test_generate_arrays_parallel(
        self,
        m_pmap: ParallelResultsIterator
    ) -> None:
        """ Test generation like above but when load_threads is greater than
        one, which should use the parallel route. """
        m_img_reader = mock.Mock(spec=ImageReader)
        # Mock image reader to return a matrix for an input something.
        m_img_reader.load_as_matrix.side_effect = lambda e: "matrix!"+e

        m_img_dg = mock.Mock(spec=ImageDescriptorGenerator)
        # Mock generator method to simply pass-through its input
        m_img_dg.generate_arrays_from_images.side_effect = \
            lambda it: ('descriptor!'+v for v in it)

        inst = ImageDescriptorGeneratorWrapper(m_img_reader, m_img_dg,
                                               image_load_threads=3)
        assert inst._image_load_threads == 3

        # pass some dummy elements (strings) and observe that each are
        # "converted" to a matrix, and that the return iterator results
        # noinspection DuplicatedCode,PyTypeChecker
        ret = list(inst._generate_arrays(['b', 'a']))  # type: ignore

        # Image reader should have been called at least once per input.
        assert m_img_reader.load_as_matrix.call_count == 2
        m_img_reader.load_as_matrix.assert_any_call('a')
        m_img_reader.load_as_matrix.assert_any_call('b')

        # Image descriptor should have been called once with an iterator of the
        # combined "matrix" outputs.
        m_img_dg.generate_arrays_from_images.assert_called_once()
        assert ret == ['descriptor!matrix!b', 'descriptor!matrix!a']

        # Parallel map should have been called with the appropriate number of
        # "cores"
        assert m_pmap.call_count == 1  # type: ignore
        m_pmap.assert_called_once_with(m_img_reader.load_as_matrix,  # type: ignore
                                       ['b', 'a'],
                                       cores=3)
