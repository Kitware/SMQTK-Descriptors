import os
import pickle
from typing import cast, Any, Dict
import unittest

import unittest.mock as mock
import numpy
import pytest

from smqtk_core.configuration import configuration_test_helper, to_config_dict
from smqtk_dataprovider import DataElement
from smqtk_dataprovider.impls.data_element.file import DataFileElement
from smqtk_dataprovider.impls.data_element.memory import DataMemoryElement

from smqtk_descriptors import DescriptorGenerator
# noinspection PyProtectedMember
from smqtk_descriptors.impls.descriptor_generator.caffe1 import (
    caffe,
    PIL,
    CaffeDescriptorGenerator,
    # Testing protected helper function
    _process_load_img_array,
)

from tests import TEST_DATA_DIR


@unittest.skipUnless(CaffeDescriptorGenerator.is_usable(),
                     reason="CaffeDescriptorGenerator is not usable in"
                            "current environment.")
class TestCaffeDesctriptorGenerator (unittest.TestCase):

    hopper_image_fp = os.path.join(TEST_DATA_DIR, 'grace_hopper.png')

    # Dummy Caffe configuration files + weights
    # - weights is actually an empty file (0 bytes), which caffe treats
    #   as random/zero values (not sure exactly what's happening, but
    #   always results in a zero-vector).
    dummy_net_topo_elem = DataFileElement(
        os.path.join(TEST_DATA_DIR, 'caffe.dummpy_network.prototxt'),
        readonly=True
    )
    dummy_caffe_model_elem = DataFileElement(
        os.path.join(TEST_DATA_DIR, 'caffe.empty_model.caffemodel'),
        readonly=True
    )
    dummy_img_mean_elem = DataFileElement(
        os.path.join(TEST_DATA_DIR, 'caffe.dummy_mean.npy'),
        readonly=True
    )

    def test_impl_findable(self) -> None:
        self.assertIn(CaffeDescriptorGenerator,
                      DescriptorGenerator.get_impls())

    def test_init_no_prototxt_no_model(self) -> None:
        """
        Test that the class fails to construct and initialize if no
        network prototext or model are provided.
        """
        with pytest.raises(AttributeError,
                           match="'NoneType' object has no attribute"):
            # noinspection PyTypeChecker
            CaffeDescriptorGenerator(
                network_prototxt=None,  # type: ignore
                network_model=None  # type: ignore
            )

    def test_init_no_model(self) -> None:
        """
        Test that the class fails to construct and initialize if only no
        prototext DataElement is provided.
        """
        with pytest.raises(AttributeError,
                           match="'NoneType' object has no attribute"):
            # noinspection PyTypeChecker
            CaffeDescriptorGenerator(
                network_prototxt=self.dummy_net_topo_elem,
                network_model=None  # type: ignore
            )

    def test_init_no_prototxt(self) -> None:
        """
        Test that the class fails to construct and initialize if only no
        model DataElement is provided.
        """
        with pytest.raises(AttributeError,
                           match="'NoneType' object has no attribute"):
            # noinspection PyTypeChecker
            CaffeDescriptorGenerator(
                network_prototxt=None,  # type: ignore
                network_model=self.dummy_caffe_model_elem
            )

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.caffe1'
                '.CaffeDescriptorGenerator._setup_network')
    def test_get_config(self, _m_cdg_setupNetwork: mock.MagicMock) -> None:
        # Mocking set_network so we don't have to worry about actually
        # initializing any caffe things for this test.
        expected_params: Dict[str, Any] = {
            'network_prototxt': DataMemoryElement(),
            'network_model': DataMemoryElement(),
            'image_mean': DataMemoryElement(),
            'return_layer': 'layer name',
            'batch_size': 777,
            'use_gpu': False,
            'gpu_device_id': 8,
            'network_is_bgr': False,
            'data_layer': 'data-other',
            'load_truncated_images': True,
            'pixel_rescale': (.2, .8),
            'input_scale': 1.5,
            'threads': 14,
        }
        # make sure that we're considering all constructor parameter
        # options
        default_params = CaffeDescriptorGenerator.get_default_config()
        assert set(default_params) == set(expected_params)
        g = CaffeDescriptorGenerator(**expected_params)

        # Shift to expecting sub-configs for DataElement params
        for key in ('network_prototxt', 'network_model', 'image_mean'):
            expected_params[key] = to_config_dict(cast(DataMemoryElement, expected_params[key]))
        assert g.get_config() == expected_params

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.caffe1'
                '.CaffeDescriptorGenerator._setup_network')
    def test_config_cycle(self, m_cdg_setup_network: mock.MagicMock) -> None:
        """
        Test being able to get an instances config and use that config to
        construct an equivalently parameterized instance. This test initializes
        all possible parameters to non-defaults.
        """
        # Mocking ``_setup_network`` so no caffe functionality is hit during
        # this test

        # When every parameter is provided.
        g1 = CaffeDescriptorGenerator(self.dummy_net_topo_elem,
                                      self.dummy_caffe_model_elem,
                                      image_mean=self.dummy_img_mean_elem,
                                      return_layer='foobar',
                                      batch_size=9,
                                      use_gpu=True,
                                      gpu_device_id=99,
                                      network_is_bgr=False,
                                      data_layer='maybe data',
                                      load_truncated_images=True,
                                      pixel_rescale=(0.2, 0.3),
                                      input_scale=8.9,
                                      threads=7)
        for inst in configuration_test_helper(g1):  # type: CaffeDescriptorGenerator
            assert inst.network_prototxt == self.dummy_net_topo_elem
            assert inst.network_model == self.dummy_caffe_model_elem
            assert inst.image_mean == self.dummy_img_mean_elem
            assert inst.return_layer == 'foobar'
            assert inst.batch_size == 9
            assert inst.use_gpu is True
            assert inst.gpu_device_id == 99
            assert inst.network_is_bgr is False
            assert inst.data_layer == 'maybe data'
            assert inst.load_truncated_images is True
            assert inst.pixel_rescale == (0.2, 0.3)
            assert inst.input_scale == 8.9
            assert inst.threads == 7

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.caffe1'
                '.CaffeDescriptorGenerator._setup_network')
    def test_config_cycle_imagemean_nonevalued(self, m_cdg_setup_network: mock.MagicMock) -> None:
        """
        Test being able to get an instances config and use that config to
        construct an equivalently parameterized instance where the second
        instance is configured with a None-valued 'image_mean' parameter.
        """
        # Mocking ``_setup_network`` so no caffe functionality is hit during
        # this test

        # Only required parameters, image_mean is None
        g1 = CaffeDescriptorGenerator(self.dummy_net_topo_elem,
                                      self.dummy_caffe_model_elem)
        g1_config = g1.get_config()
        # Modify config for g2 to pass None for image_mean
        for_g2 = dict(g1_config)
        for_g2['image_mean'] = None
        g2 = CaffeDescriptorGenerator.from_config(for_g2)
        expected_config = {
            'network_prototxt': to_config_dict(self.dummy_net_topo_elem),
            'network_model': to_config_dict(self.dummy_caffe_model_elem),
            'image_mean': None,
            'return_layer': 'fc7',
            'batch_size': 1,
            'use_gpu': False,
            'gpu_device_id': 0,
            'network_is_bgr': True,
            'data_layer': 'data',
            'load_truncated_images': False,
            'pixel_rescale': None,
            'input_scale': None,
            'threads': None,
        }
        assert g1_config == g2.get_config() == expected_config

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.caffe1'
                '.CaffeDescriptorGenerator._setup_network')
    def test_config_cycle_imagemean_nonetyped(self, m_cdg_setup_network: mock.MagicMock) -> None:
        """
        Test being able to get an instances config and use that config to
        construct an equivalently parameterized instance where the second
        instance is configured with a None-typed  'image_mean' parameter.
        """
        # Mocking ``_setup_network`` so no caffe functionality is hit during
        # this test

        # Only required parameters, image_mean is empty SMQTK configuration
        # dict
        g1 = CaffeDescriptorGenerator(self.dummy_net_topo_elem,
                                      self.dummy_caffe_model_elem)
        g1_config = g1.get_config()
        # Modify config for g2 to pass None for image_mean
        for_g2 = dict(g1_config)
        for_g2['image_mean'] = {'type': None}
        g2 = CaffeDescriptorGenerator.from_config(for_g2)
        expected_config = {
            'network_prototxt': to_config_dict(self.dummy_net_topo_elem),
            'network_model': to_config_dict(self.dummy_caffe_model_elem),
            'image_mean': None,
            'return_layer': 'fc7',
            'batch_size': 1,
            'use_gpu': False,
            'gpu_device_id': 0,
            'network_is_bgr': True,
            'data_layer': 'data',
            'load_truncated_images': False,
            'pixel_rescale': None,
            'input_scale': None,
            'threads': None,
        }
        assert g1_config == g2.get_config() == expected_config

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.caffe1'
                '.CaffeDescriptorGenerator._setup_network')
    def test_pickle_save_restore(self, m_cdg_setupNetwork: mock.MagicMock) -> None:
        # Mocking set_network so we don't have to worry about actually
        # initializing any caffe things for this test.
        expected_params: Dict[str, Any] = {
            'network_prototxt': DataMemoryElement(),
            'network_model': DataMemoryElement(),
            'image_mean': DataMemoryElement(),
            'return_layer': 'layer name',
            'batch_size': 777,
            'use_gpu': False,
            'gpu_device_id': 8,
            'network_is_bgr': False,
            'data_layer': 'data-other',
            'load_truncated_images': True,
            'pixel_rescale': (.2, .8),
            'input_scale': 1.5,
            'threads': 9,
        }
        g = CaffeDescriptorGenerator(**expected_params)
        # Initialization sets up the network on construction.
        self.assertEqual(m_cdg_setupNetwork.call_count, 1)

        g_pickled = pickle.dumps(g, -1)
        g2 = pickle.loads(g_pickled)
        # Network should be setup for second class class just like in
        # initial construction.
        self.assertEqual(m_cdg_setupNetwork.call_count, 2)

        self.assertIsInstance(g2, CaffeDescriptorGenerator)
        self.assertEqual(g.get_config(), g2.get_config())

    @mock.patch('smqtk_descriptors.impls.descriptor_generator.caffe1'
                '.CaffeDescriptorGenerator._setup_network')
    def test_invalid_datatype(self, _m_cdg_setupNetwork: mock.MagicMock) -> None:
        # Test that a data element with an incorrect content type for this
        # implementation raises an exception.
        # TODO: This probably doesn't need to exist because this is mostly
        #       testing the parent class functionality that should already be
        #       covered by parent class unit tests.

        # Passing purposefully bag constructor parameters and ignoring
        # Caffe network setup (above mocking).
        m_data = mock.MagicMock(spec=DataElement)
        # noinspection PyTypeChecker
        g = CaffeDescriptorGenerator(m_data, m_data, None)
        bad_element = DataFileElement(
            os.path.join(TEST_DATA_DIR, 'test_file.dat'), readonly=True
        )
        # implementation of _generate_arrays checks for self.network value, so
        # let's set a dummy value to get to the other error condition we want
        # to actually test.
        g.network = mock.MagicMock(spec=caffe.Net)
        with pytest.raises(ValueError):
            list(g.generate_arrays([bad_element]))

    def test_process_load_img(self) -> None:
        # using image shape, meaning no transformation should occur
        test_data_layer = 'data'
        test_transformer = \
            caffe.io.Transformer({test_data_layer: (1, 3, 600, 512)})

        hopper_elem = DataFileElement(self.hopper_image_fp, readonly=True)
        a_expected = numpy.asarray(PIL.Image.open(self.hopper_image_fp),
                                   numpy.float32)
        a = _process_load_img_array((
            hopper_elem, test_transformer, test_data_layer, False, None
        ))
        numpy.testing.assert_allclose(a[1], a_expected)

    def test_generate_arrays_dummy_model(self) -> None:
        # Caffe dummy network interaction test Grace Hopper image)

        # Construct network with an empty model just to see that our
        # interaction with the Caffe API is successful. We expect a
        # zero-valued descriptor vector.
        g = CaffeDescriptorGenerator(self.dummy_net_topo_elem,
                                     self.dummy_caffe_model_elem,
                                     self.dummy_img_mean_elem,
                                     return_layer='fc', use_gpu=False)
        d_list = list(g._generate_arrays(
            [DataFileElement(self.hopper_image_fp, readonly=True)]
        ))
        assert len(d_list) == 1
        d = d_list[0]
        self.assertAlmostEqual(d.sum(), 0., 12)

    def test_generate_arrays_no_data(self) -> None:
        """ Test that generation method correctly returns an empty iterable
        when no data is passed. """
        g = CaffeDescriptorGenerator(self.dummy_net_topo_elem,
                                     self.dummy_caffe_model_elem,
                                     self.dummy_img_mean_elem,
                                     return_layer='fc', use_gpu=False)
        r = list(g._generate_arrays([]))
        assert r == []
