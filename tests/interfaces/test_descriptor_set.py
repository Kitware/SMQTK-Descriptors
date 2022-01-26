from typing import Any, Dict, Generator, Hashable, Iterable, Iterator, Tuple
import unittest
import unittest.mock as mock

from smqtk_descriptors import DescriptorElement, DescriptorSet


class DummyDescriptorSet (DescriptorSet):

    def get_config(self) -> Dict[str, Any]: ...

    def get_descriptor(self, uuid: Hashable) -> DescriptorElement: ...

    def get_many_descriptors(self, uuids: Iterable[Hashable]) -> Iterator[DescriptorElement]: ...

    def keys(self) -> Iterator[Hashable]: ...

    def items(self) -> Iterator[Tuple[Hashable, DescriptorElement]]: ...

    def descriptors(self) -> Iterator[DescriptorElement]: ...

    def remove_many_descriptors(self, uuids: Iterable[Hashable]) -> None: ...

    def has_descriptor(self, uuid: Hashable) -> bool: ...

    def add_many_descriptors(self, descriptors: Iterable[DescriptorElement]) -> None: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...

    def remove_descriptor(self, uuid: Hashable) -> None: ...

    def add_descriptor(self, descriptor: DescriptorElement) -> None: ...


class TestDescriptorSetAbstract (unittest.TestCase):

    def test_len(self) -> None:
        di = DummyDescriptorSet()
        # noinspection PyTypeHints
        di.count = mock.Mock(return_value=100)  # type: ignore
        self.assertEqual(len(di), 100)
        di.count.assert_called_once_with()

    def test_get_item(self) -> None:
        di = DummyDescriptorSet()
        # noinspection PyTypeHints
        di.get_descriptor = mock.Mock(return_value='foo')  # type: ignore
        self.assertEqual(di['some_key'], 'foo')
        di.get_descriptor.assert_called_once_with('some_key')

    def test_del_item(self) -> None:
        di = DummyDescriptorSet()
        # noinspection PyTypeHints
        di.remove_descriptor = mock.Mock()  # type: ignore

        del di['foo']
        di.remove_descriptor.assert_called_once_with('foo')

    def test_iter(self) -> None:
        # Iterating over a DescriptorSet should yield the descriptor elements
        di = DummyDescriptorSet()

        def dumb_iterator() -> Generator[int, None, None]:
            for _i in range(3):
                yield _i

        # noinspection PyTypeHints
        di.descriptors = mock.Mock(side_effect=dumb_iterator)  # type: ignore

        for i, v in enumerate(iter(di)):
            self.assertEqual(i, v)
        self.assertEqual(list(di), [0, 1, 2])
        self.assertEqual(tuple(di), (0, 1, 2))
        self.assertEqual(di.descriptors.call_count, 3)

    @mock.patch("smqtk_descriptors.interfaces.descriptor_set.DescriptorElement"
                ".get_many_vectors", wraps=DescriptorElement.get_many_vectors)
    def test_get_many_vectors_empty(self, m_de_gmv: mock.MagicMock) -> None:
        """ Test that no vectors are returned when no UIDs are provided. """
        inst = DummyDescriptorSet()
        # noinspection PyTypeHints
        inst.get_many_descriptors = mock.Mock(return_value=[])  # type: ignore
        r = inst.get_many_vectors([])
        assert r == []
        m_de_gmv.assert_called_once_with([])
