import os
import pickle
from typing import Any, Dict
import unittest

import unittest.mock as mock
import numpy as np

from smqtk_core.configuration import configuration_test_helper, make_default_config
from smqtk_dataprovider.impls.data_element.file import DataFileElement
from smqtk_image_io import ImageReader
from smqtk_image_io.impls.image_reader.pil_io import PilImageReader

from smqtk_descriptors import DescriptorGenerator
# noinspection PyProtectedMember
from smqtk_descriptors.impls.descriptor_generator.pytorch import (
    TorchModuleDescriptorGenerator,
    Resnet50SequentialTorchDescriptorGenerator,
    AlignedReIDResNet50TorchDescriptorGenerator,
)

from tests import TEST_DATA_DIR


@unittest.skipUnless(TorchModuleDescriptorGenerator.is_usable(),
                     reason="TorchModuleDescriptorGenerator is not usable in"
                            "current environment.")
class TestTorchDescriptorGenerator (unittest.TestCase):

    hopper_image_fp = os.path.join(TEST_DATA_DIR, 'grace_hopper.png')

    dummy_image_reader = PilImageReader()

    def test_impl_findable(self) -> None:
        self.assertIn(Resnet50SequentialTorchDescriptorGenerator,
                      DescriptorGenerator.get_impls())

        self.assertIn(AlignedReIDResNet50TorchDescriptorGenerator,
                      DescriptorGenerator.get_impls())

    def test_get_config(self) -> None:
        expected_params: Dict[str, Any] = {
            'image_reader': make_default_config(ImageReader.get_impls()),
            'image_load_threads': 1,
            'weights_filepath': None,
            'image_tform_threads': 1,
            'batch_size': 32,
            'use_gpu': False,
            'cuda_device': None,
            'normalize': None,
            'iter_runtime': False,
            'global_average_pool': False
        }
        # make sure that we're considering all constructor parameter
        # options
        default_params = TorchModuleDescriptorGenerator.get_default_config()
        assert set(default_params) == set(expected_params)

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.pytorch'
                '.TorchModuleDescriptorGenerator._ensure_module')
    def test_config_cycle(self, m_tdg_ensure_module: mock.MagicMock) -> None:
        """
        Test being able to get an instances config and use that config to
        construct an equivalently parameterized instance. This test initializes
        all possible parameters to non-defaults.
        """
        # When every parameter is provided.
        g1 = Resnet50SequentialTorchDescriptorGenerator(self.dummy_image_reader,
                                                        image_load_threads=2,
                                                        weights_filepath="cool_filepath",
                                                        image_tform_threads=2,
                                                        batch_size=64,
                                                        use_gpu=True,
                                                        cuda_device=1,
                                                        normalize=1.0,
                                                        iter_runtime=True,
                                                        global_average_pool=True)
        for inst_g1 in configuration_test_helper(g1):
            assert isinstance(inst_g1.image_reader, type(self.dummy_image_reader))
            assert inst_g1.image_load_threads == 2
            assert inst_g1.weights_filepath == "cool_filepath"
            assert inst_g1.image_tform_threads == 2
            assert inst_g1.batch_size == 64
            assert inst_g1.use_gpu is True
            assert inst_g1.cuda_device == 1
            assert inst_g1.normalize == 1.0
            assert inst_g1.iter_runtime is True
            assert inst_g1.global_average_pool is True

        # Repeat for AlignedReIDResNet50
        g2 = AlignedReIDResNet50TorchDescriptorGenerator(self.dummy_image_reader,
                                                         image_load_threads=2,
                                                         weights_filepath="cool_filepath",
                                                         image_tform_threads=2,
                                                         batch_size=64,
                                                         use_gpu=True,
                                                         cuda_device=1,
                                                         normalize=1.0,
                                                         iter_runtime=True,
                                                         global_average_pool=True)
        for inst_g2 in configuration_test_helper(g2):
            assert isinstance(inst_g2.image_reader, type(self.dummy_image_reader))
            assert inst_g2.image_load_threads == 2
            assert inst_g2.weights_filepath == "cool_filepath"
            assert inst_g2.image_tform_threads == 2
            assert inst_g2.batch_size == 64
            assert inst_g2.use_gpu is True
            assert inst_g2.cuda_device == 1
            assert inst_g2.normalize == 1.0
            assert inst_g2.iter_runtime is True
            assert inst_g2.global_average_pool is True

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.pytorch'
                '.TorchModuleDescriptorGenerator._ensure_module')
    def test_pickle_save_restore(self, m_tdg_ensure_module: mock.MagicMock) -> None:
        expected_params: Dict[str, Any] = {
            'image_reader': self.dummy_image_reader,
            'image_load_threads': 1,
            'weights_filepath': None,
            'image_tform_threads': 1,
            'batch_size': 32,
            'use_gpu': False,
            'cuda_device': None,
            'normalize': None,
            'iter_runtime': False,
            'global_average_pool': False
        }
        g = Resnet50SequentialTorchDescriptorGenerator(**expected_params)
        # Initialization sets up the network on construction.
        self.assertEqual(m_tdg_ensure_module.call_count, 1)

        g_pickled = pickle.dumps(g, -1)
        g2 = pickle.loads(g_pickled)

        # Network should be setup for second class class just like in
        # initial construction.
        self.assertEqual(m_tdg_ensure_module.call_count, 2)

        self.assertIsInstance(g2, Resnet50SequentialTorchDescriptorGenerator)
        self.assertEqual(g.get_config(), g2.get_config())

        # Repeat for AlignedReIDResNet50
        g3 = AlignedReIDResNet50TorchDescriptorGenerator(**expected_params)
        self.assertEqual(m_tdg_ensure_module.call_count, 3)

        g3_pickled = pickle.dumps(g3, -1)
        g4 = pickle.loads(g3_pickled)

        self.assertEqual(m_tdg_ensure_module.call_count, 4)

        self.assertIsInstance(g3, AlignedReIDResNet50TorchDescriptorGenerator)
        self.assertEqual(g3.get_config(), g4.get_config())

    def test_generate_arrays(self) -> None:
        g1 = Resnet50SequentialTorchDescriptorGenerator(self.dummy_image_reader)
        d_list_g1 = list(g1._generate_arrays(
            [DataFileElement(self.hopper_image_fp, readonly=True)]
        ))
        assert len(d_list_g1) == 1
        d_resnet_seq = d_list_g1[0]

        g2 = AlignedReIDResNet50TorchDescriptorGenerator(self.dummy_image_reader)
        d_list_g2 = list(g2._generate_arrays(
            [DataFileElement(self.hopper_image_fp, readonly=True)]
        ))
        assert len(d_list_g2) == 1
        d_aligned_reid = d_list_g2[0]

        # Check that the descriptors generated by both implementations are close
        np.testing.assert_allclose(d_resnet_seq, d_aligned_reid, 1e-4)

    def test_generate_arrays_no_data(self) -> None:
        """ Test that generation method correctly returns an empty iterable
        when no data is passed. """
        g1 = Resnet50SequentialTorchDescriptorGenerator(self.dummy_image_reader)
        r1 = list(g1._generate_arrays([]))
        assert r1 == []

        # Repeat for AlignedReIDResNet50
        g2 = AlignedReIDResNet50TorchDescriptorGenerator(self.dummy_image_reader)
        r2 = list(g2._generate_arrays([]))
        assert r2 == []
