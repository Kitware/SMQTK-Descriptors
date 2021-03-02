import logging
import pickle
from typing import Any, Dict, Hashable, Iterable, Iterator, Optional, Tuple, Type, TypeVar

from smqtk_core.configuration import from_config_dict, make_default_config, to_config_dict
from smqtk_core.dict import merge_dict
from smqtk_dataprovider import DataElement
from smqtk_dataprovider.utils import SimpleTimer
from smqtk_descriptors import DescriptorElement, DescriptorSet


LOG = logging.getLogger(__name__)
# Type variable for Configurable-inheriting types.
MDS = TypeVar("MDS", bound="MemoryDescriptorSet")


class MemoryDescriptorSet (DescriptorSet):
    """
    In-memory descriptor index with file caching.

    Stored descriptor elements are all held in memory in a uuid-to-element
    dictionary (hash table).

    If the path to a file cache is provided, it is loaded at construction if it
    exists. When elements are added to the index, the in-memory table is dumped
    to the cache.
    """

    @classmethod
    def is_usable(cls) -> bool:
        """
        Check whether this class is available for use.

        :return: Boolean determination of whether this implementation is usable.

        """
        # no dependencies
        return True

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        By default, we observe what this class's constructor takes as arguments,
        turning those argument names into configuration dictionary keys. If any
        of those arguments have defaults, we will add those values into the
        configuration dictionary appropriately. The dictionary returned should
        only contain JSON compliant value types.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.

        """
        c = super(MemoryDescriptorSet, cls).get_default_config()
        c['cache_element'] = make_default_config(DataElement.get_impls())
        return c

    @classmethod
    def from_config(
        cls: Type[MDS],
        config_dict: Dict,
        merge_default: bool = True
    ) -> MDS:
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.

        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.

        :return: Constructed instance from the provided config.

        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        # Optionally construct cache element from sub-config.
        if config_dict['cache_element'] \
                and config_dict['cache_element']['type']:
            e = from_config_dict(config_dict['cache_element'],
                                 DataElement.get_impls())
            config_dict['cache_element'] = e
        else:
            config_dict['cache_element'] = None

        return super(MemoryDescriptorSet, cls).from_config(config_dict, False)

    def __init__(
        self,
        cache_element: Optional[DataElement] = None,
        pickle_protocol: int = -1
    ):
        """
        Initialize a new in-memory descriptor index, or reload one from a
        cache.

        :param cache_element: Optional data element cache, loading an existing
            index if the element has bytes. If the given element is writable,
             new descriptors added to this index are cached to the element.

        :param pickle_protocol: Pickling protocol to use when serializing index
            table to the optionally provided, writable cache element. We will
            use -1 by default (latest version, probably a binary form).

        """
        super(MemoryDescriptorSet, self).__init__()

        # Mapping of descriptor UUID to the DescriptorElement instance.
        self._table: Dict[Hashable, DescriptorElement] = {}
        # Record of optional file cache we're using
        self.cache_element = cache_element
        self.pickle_protocol = pickle_protocol

        if cache_element and not cache_element.is_empty():
            LOG.debug(f"Loading cached descriptor index table from "
                      f"{cache_element.__class__.__name__} element.")
            self._table = pickle.loads(cache_element.get_bytes())
            assert isinstance(self._table, dict), "Loaded cache structure was not a dictionary type!"

    def get_config(self) -> Dict[str, Any]:
        c = merge_dict(self.get_default_config(), {
            "pickle_protocol": self.pickle_protocol,
        })
        if self.cache_element:
            merge_dict(c['cache_element'],
                       to_config_dict(self.cache_element))
        return c

    def cache_table(self) -> None:
        if self.cache_element and self.cache_element.writable():
            with SimpleTimer("Caching descriptor table", LOG.debug):
                self.cache_element.set_bytes(pickle.dumps(self._table,
                                                          self.pickle_protocol))

    def count(self) -> int:
        return len(self._table)

    def clear(self) -> None:
        """
        Clear this descriptor index's entries.
        """
        self._table = {}
        self.cache_table()

    def has_descriptor(self, uuid: Hashable) -> bool:
        """
        Check if a DescriptorElement with the given UUID exists in this index.

        :param uuid: UUID to query for

        :return: True if a DescriptorElement with the given UUID exists in this
            index, or False if not.

        """
        return uuid in self._table

    def add_descriptor(self, descriptor: DescriptorElement) -> None:
        """
        Add a descriptor to this index.

        Adding the same descriptor multiple times should not add multiple
        copies of the descriptor in the index.

        :param descriptor: Descriptor to index.
        """
        self._inner_add_descriptor(descriptor, no_cache=False)

    def _inner_add_descriptor(
        self,
        descriptor: DescriptorElement,
        no_cache: bool = False
    ) -> None:
        """
        Internal adder with the additional option to trigger caching or not.

        :param descriptor: Descriptor to index.
        :param no_cache: Do not cache the internal table if a file cache was
            provided. This would be used if adding many descriptors at a time,
            preventing a file write for every individual descriptor added.
        """
        self._table[descriptor.uuid()] = descriptor
        if not no_cache:
            self.cache_table()

    def add_many_descriptors(self, descriptors: Iterable[DescriptorElement]) -> None:
        """
        Add multiple descriptors at one time.

        :param descriptors: Iterable of descriptor instances to add to this
            index.

        """
        added_something = False
        for d in descriptors:
            # using no-cache so we don't trigger multiple file writes
            self._inner_add_descriptor(d, no_cache=True)
            added_something = True
        if added_something:
            self.cache_table()

    def get_descriptor(self, uuid: Hashable) -> DescriptorElement:
        """
        Get the descriptor in this index that is associated with the given UUID.

        :param uuid: UUID of the DescriptorElement to get.

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        :return: DescriptorElement associated with the queried UUID.

        """
        return self._table[uuid]

    def get_many_descriptors(self, uuids: Iterable[Hashable]) -> Iterator[DescriptorElement]:
        """
        Get an iterator over descriptors associated to given descriptor UUIDs.

        :param uuids: Iterable of descriptor UUIDs to query for.

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        :return: Iterator of descriptors associated to given uuid values.

        """
        for uid in uuids:
            yield self._table[uid]

    def remove_descriptor(self, uuid: Hashable) -> None:
        """
        Remove a descriptor from this index by the given UUID.

        :param uuid: UUID of the DescriptorElement to remove.

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this index.

        """
        self._inner_remove_descriptor(uuid, no_cache=False)

    def _inner_remove_descriptor(
        self,
        uuid: Hashable,
        no_cache: bool = False
    ) -> None:
        """
        Internal remover with the additional option to trigger caching or not.

        :param uuid: UUID of the DescriptorElement to remove.
        :param no_cache: Do not cache the internal table if a file cache was
            provided. This would be used if adding many descriptors at a time,
            preventing a file write for every individual descriptor added.
        """
        del self._table[uuid]
        if not no_cache:
            self.cache_table()

    def remove_many_descriptors(self, uuids: Iterable[Hashable]) -> None:
        """
        Remove descriptors associated to given descriptor UUIDs from this
        index.

        :param uuids: Iterable of descriptor UUIDs to remove.

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this index.

        """
        for uid in uuids:
            # using no-cache so we don't trigger multiple file writes
            self._inner_remove_descriptor(uid, no_cache=True)
        self.cache_table()

    def iterkeys(self) -> Iterator[Hashable]:
        return iter(self._table.keys())

    def iterdescriptors(self) -> Iterator[DescriptorElement]:
        return iter(self._table.values())

    def iteritems(self) -> Iterator[Tuple[Hashable, DescriptorElement]]:
        return iter(self._table.items())
