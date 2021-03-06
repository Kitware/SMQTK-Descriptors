import pickle
import unittest.mock as mock
import unittest

import numpy

from smqtk_core.configuration import configuration_test_helper
from smqtk_descriptors.impls.descriptor_element.file import DescriptorFileElement


class TestDescriptorFileElement (unittest.TestCase):

    def test_configuration(self) -> None:
        """ Test instance standard configuration """
        inst = DescriptorFileElement('test', 'abcd',
                                     save_dir='/some/path/somewhere',
                                     subdir_split=4)
        for i in configuration_test_helper(inst, {'type_str', 'uuid'},
                                           ('test', 'abcd')):
            assert i._save_dir == '/some/path/somewhere'
            assert i._subdir_split == 4

    def test_vec_filepath_generation(self) -> None:
        d = DescriptorFileElement('test', 'abcd', '/base', 4)
        self.assertEqual(d._vec_filepath,
                         '/base/a/b/c/test.abcd.vector.npy')

        d = DescriptorFileElement('test', 'abcd', '/base', 2)
        self.assertEqual(d._vec_filepath,
                         '/base/ab/test.abcd.vector.npy')

        d = DescriptorFileElement('test', 'abcd', '/base', 1)
        self.assertEqual(d._vec_filepath,
                         '/base/test.abcd.vector.npy')

        d = DescriptorFileElement('test', 'abcd', '/base', 0)
        self.assertEqual(d._vec_filepath,
                         '/base/test.abcd.vector.npy')

        d = DescriptorFileElement('test', 'abcd', '/base')
        self.assertEqual(d._vec_filepath,
                         '/base/test.abcd.vector.npy')

    def test_serialization(self) -> None:
        # Test that an instance can be serialized and deserialized via pickle
        # successfully.
        ex_type = 'test'
        ex_uid = 12345
        ex_save_dir = 'some-dir'
        ex_split = 5
        e1 = DescriptorFileElement(ex_type, ex_uid, ex_save_dir, ex_split)

        # pickle dump and load into a new copy
        e2: DescriptorFileElement = pickle.loads(pickle.dumps(e1))
        # Make sure the two have the same attributes, including base descriptor
        # element things.
        self.assertEqual(e1.type(), e2.type())
        self.assertEqual(e1.uuid(), e2.uuid())
        self.assertEqual(e1._save_dir, e2._save_dir)
        self.assertEqual(e1._subdir_split, e2._subdir_split)
        self.assertEqual(e1._vec_filepath, e2._vec_filepath)

    @mock.patch('smqtk_descriptors.impls.descriptor_element.file.numpy.save')
    @mock.patch('smqtk_descriptors.impls.descriptor_element.file.safe_create_dir')
    def test_vector_set(self, mock_scd: mock.MagicMock, mock_save: mock.MagicMock) -> None:
        d = DescriptorFileElement('test', 1234, '/base', 4)
        self.assertEqual(d._vec_filepath,
                         '/base/1/2/3/test.1234.vector.npy')

        v = numpy.zeros(16)
        d.set_vector(v)
        mock_scd.assert_called_with('/base/1/2/3')
        mock_save.assert_called_with('/base/1/2/3/test.1234.vector.npy', v)

    @mock.patch('smqtk_descriptors.impls.descriptor_element.file.numpy.load')
    def test_vector_get(self, mock_load: mock.MagicMock) -> None:
        d = DescriptorFileElement('test', 1234, '/base', 4)
        self.assertFalse(d.has_vector())
        self.assertIs(d.vector(), None)

        # noinspection PyTypeHints
        d.has_vector = mock.Mock(return_value=True)  # type: ignore
        self.assertTrue(d.has_vector())
        v = numpy.zeros(16)
        mock_load.return_value = v
        numpy.testing.assert_equal(d.vector(), v)
