import abc
from typing import Hashable, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from smqtk_core import Configurable, Pluggable
from smqtk_descriptors import DescriptorElement


class DescriptorSet (Configurable, Pluggable):
    """
    Index of descriptors, keyed and query-able by descriptor UUID.

    Note that these indexes do not use the descriptor type strings. Thus, if
    a set of descriptors has multiple elements with the same UUID, but
    different type strings, they will bash each other in these indexes. In such
    a case, when dealing with descriptors for different generators, it is
    advisable to use multiple indices.

    """

    def __delitem__(self, uuid: Hashable) -> None:
        self.remove_descriptor(uuid)

    def __getitem__(self, uuid: Hashable) -> DescriptorElement:
        return self.get_descriptor(uuid)

    def __iter__(self) -> Iterator[DescriptorElement]:
        return self.iterdescriptors()

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, item: DescriptorElement) -> bool:
        if isinstance(item, DescriptorElement):
            # Testing for UUID inclusion since element hash based on UUID
            # value.
            return self.has_descriptor(item.uuid())
        return False

    def get_many_vectors(self, uuids: Iterable[Hashable]) -> List[Optional[np.ndarray]]:
        """
        Get underlying vectors of descriptors associated with given uuids.

        :param uuids: Iterable of descriptor UUIDs to query for.

        :raises: KeyError: When there is not a descriptor in this set for one
            or more input UIDs.

        :return: List of vectors for descriptors associated with given uuid
            values.

        """
        return DescriptorElement.get_many_vectors(
            self.get_many_descriptors(uuids)
        )

    @abc.abstractmethod
    def count(self) -> int:
        """
        :return: Number of descriptor elements stored in this index.
        """

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clear this descriptor index's entries.
        """

    @abc.abstractmethod
    def has_descriptor(self, uuid: Hashable) -> bool:
        """
        Check if a DescriptorElement with the given UUID exists in this index.

        :param uuid: UUID to query for

        :return: True if a DescriptorElement with the given UUID exists in this
            index, or False if not.

        """

    @abc.abstractmethod
    def add_descriptor(self, descriptor: DescriptorElement) -> None:
        """
        Add a descriptor to this index.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the index (based on UUID). Added descriptors
        overwrite indexed descriptors based on UUID.

        :param descriptor: Descriptor to index.

        """

    @abc.abstractmethod
    def add_many_descriptors(self, descriptors: Iterable[DescriptorElement]) -> None:
        """
        Add multiple descriptors at one time.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the index (based on UUID). Added descriptors
        overwrite indexed descriptors based on UUID.

        :param descriptors: Iterable of descriptor instances to add to this
            index.

        """

    @abc.abstractmethod
    def get_descriptor(self, uuid: Hashable) -> DescriptorElement:
        """
        Get the descriptor in this index that is associated with the given UUID.

        :param uuid: UUID of the DescriptorElement to get.

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        :return: DescriptorElement associated with the queried UUID.

        """

    @abc.abstractmethod
    def get_many_descriptors(self, uuids: Iterable[Hashable]) -> Iterator[DescriptorElement]:
        """
        Get an iterator over descriptors associated to given descriptor UUIDs.

        :param uuids: Iterable of descriptor UUIDs to query for.

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        :return: Iterator of descriptors associated to given uuid values.

        """

    @abc.abstractmethod
    def remove_descriptor(self, uuid: Hashable) -> None:
        """
        Remove a descriptor from this index by the given UUID.

        :param uuid: UUID of the DescriptorElement to remove.

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        """

    @abc.abstractmethod
    def remove_many_descriptors(self, uuids: Iterable[Hashable]) -> None:
        """
        Remove descriptors associated to given descriptor UUIDs from this index.

        :param uuids: Iterable of descriptor UUIDs to remove.

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        """

    @abc.abstractmethod
    def iterkeys(self) -> Iterator[Hashable]:
        """
        Return an iterator over indexed descriptor keys, which are their UUIDs.
        """

    @abc.abstractmethod
    def iterdescriptors(self) -> Iterator[DescriptorElement]:
        """
        Return an iterator over indexed descriptor element instances.
        """

    @abc.abstractmethod
    def iteritems(self) -> Iterator[Tuple[Hashable, DescriptorElement]]:
        """
        Return an iterator over indexed descriptor key and instance pairs.
        """

    def keys(self) -> Iterator[Hashable]:
        """ alias for iterkeys """
        return self.iterkeys()

    def items(self) -> Iterator[Tuple[Hashable, DescriptorElement]]:
        """ alias for iteritems """
        return self.iteritems()
