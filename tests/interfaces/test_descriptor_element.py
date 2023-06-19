from typing import Any, Dict, Optional
import unittest.mock as mock
import unittest

import numpy

from smqtk_descriptors import DescriptorElement


class DummyDescriptorElement (DescriptorElement):

    def get_config(self) -> Dict[str, Any]: pass  # type: ignore

    def set_vector(self, new_vec: numpy.ndarray) -> "DescriptorElement": pass  # type: ignore

    def has_vector(self) -> bool: pass  # type: ignore

    def vector(self) -> Optional[numpy.ndarray]: pass  # type: ignore


class TestDescriptorElementAbstract (unittest.TestCase):

    def test_init(self) -> None:
        expected_uuid = 'some uuid'
        de = DummyDescriptorElement(expected_uuid)
        self.assertEqual(de.uuid(), expected_uuid)

    def test_equality(self) -> None:
        de1 = DummyDescriptorElement('u1')
        de2 = DummyDescriptorElement('u2')
        # noinspection PyTypeHints
        de1.vector = de2.vector = mock.Mock(return_value=numpy.random.randint(0, 10, 10))  # type: ignore

        self.assertTrue(de1 == de1)
        self.assertTrue(de2 == de2)
        self.assertTrue(de1 == de2)
        self.assertFalse(de1 != de2)

    def test_nonEquality_diffInstance(self) -> None:
        # diff instance
        de = DummyDescriptorElement('a')
        self.assertFalse(de == 'string')
        self.assertTrue(de != 'string')

    def test_nonEquality_diffVectors(self) -> None:
        # different vectors (same size)
        v1 = numpy.random.randint(0, 10, 10)
        v2 = numpy.random.randint(0, 10, 10)

        d1 = DummyDescriptorElement('a')
        # noinspection PyTypeHints
        d1.vector = mock.Mock(return_value=v1)  # type: ignore

        d2 = DummyDescriptorElement('a')
        # noinspection PyTypeHints
        d2.vector = mock.Mock(return_value=v2)  # type: ignore

        self.assertFalse(d1 == d2)
        self.assertTrue(d1 != d2)

    def test_nonEquality_diffVectorSize(self) -> None:
        # different sized vectors
        v1 = numpy.random.randint(0, 10, 10)
        v2 = numpy.random.randint(0, 10, 100)

        d1 = DummyDescriptorElement('a')
        # noinspection PyTypeHints
        d1.vector = mock.Mock(return_value=v1)  # type: ignore

        d2 = DummyDescriptorElement('a')
        # noinspection PyTypeHints
        d2.vector = mock.Mock(return_value=v2)  # type: ignore

        self.assertFalse(d1 == d2)
        self.assertTrue(d1 != d2)

    def test_get_many_vectors(self) -> None:
        v1 = numpy.random.randint(0, 10, 10)
        v2 = numpy.random.randint(0, 10, 100)

        d1 = DummyDescriptorElement('a')
        # noinspection PyTypeHints
        d1.vector = mock.Mock(return_value=v1)  # type: ignore

        d2 = DummyDescriptorElement('b')
        # noinspection PyTypeHints
        d2.vector = mock.Mock(return_value=v2)  # type: ignore

        retrieved_vectors = DummyDescriptorElement.get_many_vectors([d1, d2])
        for retrieved, expected in zip(retrieved_vectors, [v1, v2]):
            numpy.testing.assert_array_equal(retrieved, expected)  # type: ignore

    def test_hash(self) -> None:
        # Hash of a descriptor element is solely based on the UUID value of
        # that element.
        uuid1 = 'some uuid'
        de1 = DummyDescriptorElement(uuid1)

        uuid2 = 'some uuid'
        de2 = DummyDescriptorElement(uuid2)

        self.assertEqual(hash(de1), hash(uuid1))
        self.assertEqual(hash(de2), hash(uuid2))
        self.assertEqual(hash(de1), hash(de2))

    def test_getstate(self) -> None:
        expected_uid = 'a'
        e = DummyDescriptorElement(expected_uid)
        state = e.__getstate__()
        self.assertDictEqual(
            state,
            {
                '_uuid': expected_uid,
            }
        )

    def test_setstate(self) -> None:
        # Intentionally bad input types.
        # noinspection PyTypeChecker
        e = DummyDescriptorElement(None)
        self.assertIsNone(e._uuid)

        expected_uid = 'a'
        state = {
            '_uuid': expected_uid,
        }
        e.__setstate__(state)
        self.assertEqual(e._uuid, expected_uid)
