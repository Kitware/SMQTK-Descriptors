from typing import Any, Dict, Hashable, Optional
import unittest

import numpy

from smqtk_descriptors import DescriptorElement, DescriptorElementFactory
from smqtk_descriptors.impls.descriptor_element.memory import DescriptorMemoryElement


class DummyElementImpl (DescriptorElement):

    def __init__(self, uuid: Hashable, *args: Any, **kwds: Any):
        super(DummyElementImpl, self).__init__(uuid)
        self.args = args
        self.kwds = kwds

    def set_vector(self, new_vec: numpy.ndarray) -> "DummyElementImpl":
        return self

    def has_vector(self) -> bool:
        pass

    def vector(self) -> Optional[numpy.ndarray]:
        pass

    def get_config(self) -> Dict[str, Any]:
        pass


class TestDescriptorElemFactory (unittest.TestCase):

    def test_no_params(self) -> None:
        test_params: Dict[str, Any] = {}

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        expected_uuid = 'uuid'
        expected_args = ()
        expected_kwds: Dict[str, Any] = {}

        # Should construct a new DEI instance under they hood somewhere
        r = factory.new_descriptor(expected_uuid)

        assert isinstance(r, DummyElementImpl)
        self.assertEqual(r._uuid, expected_uuid)
        self.assertEqual(r.args, expected_args)
        self.assertEqual(r.kwds, expected_kwds)

    def test_with_params(self) -> None:
        v = numpy.random.randint(0, 10, 10)
        test_params = {
            'p1': 'some dir',
            'vec': v
        }

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        ex_uuid = 'uuid'
        ex_args = ()
        ex_kwds = test_params
        # Should construct a new DEI instance under they hood somewhere
        r = factory.new_descriptor(ex_uuid)

        assert isinstance(r, DummyElementImpl)
        self.assertEqual(r._uuid, ex_uuid)
        self.assertEqual(r.args, ex_args)
        self.assertEqual(r.kwds, ex_kwds)

    def test_call(self) -> None:
        # Same as `test_with_params` but using __call__ entry point
        v = numpy.random.randint(0, 10, 10)
        test_params = {
            'p1': 'some dir',
            'vec': v
        }

        factory = DescriptorElementFactory(DummyElementImpl, test_params)

        ex_uuid = 'uuid'
        ex_args = ()
        ex_kwds = test_params
        # Should construct a new DEI instance under they hood somewhere
        r = factory(ex_uuid)

        assert isinstance(r, DummyElementImpl)
        self.assertEqual(r._uuid, ex_uuid)
        self.assertEqual(r.args, ex_args)
        self.assertEqual(r.kwds, ex_kwds)

    def test_configuration(self) -> None:
        c = DescriptorElementFactory.get_default_config()
        self.assertIsNone(c['type'])
        dme_key = 'smqtk_descriptors.impls.descriptor_element.memory.DescriptorMemoryElement'
        self.assertIn(dme_key, c)

        c['type'] = dme_key
        factory = DescriptorElementFactory.from_config(c)
        self.assertEqual(factory._d_type.__name__,
                         DescriptorMemoryElement.__name__)
        self.assertEqual(factory._d_type_config, {})

        d = factory.new_descriptor('foo')
        self.assertEqual(d.uuid(), 'foo')

    def test_get_config(self) -> None:
        """
        We should be able to get the configuration of the current factory.
        This should look like the same as the
        """
        test_params = {
            'p1': 'some dir',
            'vec': 1
        }
        dummy_key = f"{__name__}.{DummyElementImpl.__name__}"
        factory = DescriptorElementFactory(DummyElementImpl, test_params)
        factory_config = factory.get_config()
        assert factory_config == {"type": dummy_key,
                                  dummy_key: test_params}
