import time
from typing import Any, Dict, Hashable, Mapping, Optional

import numpy

from smqtk_descriptors import DescriptorElement


# Try to import required module
try:
    import solr  # type: ignore
except ImportError:
    solr = None


class SolrDescriptorElement (DescriptorElement):  # lgtm [py/missing-equals]
    """
    Descriptor element that uses a Solr instance as the backend storage medium.

    Fields where data is stored in the Solr documents are specified at
    construction time. We additionally set the ``id`` field to a string UUID.
    ``id`` is set because it is a common, required field for unique
    identification of documents. The value set to the ``id`` field is
    reproducible from this object's key attributes.

    """

    @classmethod
    def is_usable(cls) -> bool:
        return solr is not None

    def __init__(
        self,
        uuid: Hashable,
        solr_conn_addr: str,
        uuid_field: str,
        vector_field: str,
        timestamp_field: str,
        timeout: int = 10,
        persistent_connection: bool = False,
        commit_on_set: bool = True
    ):
        """
        Initialize a new Solr-stored descriptor element.

        :param uuid: Unique ID reference of the descriptor.
        :param solr_conn_addr: HTTP(S) address for the Solr index to use
        :param uuid_field: Solr index field to store descriptor UUID string
            value in.
        :param vector_field: Solr index field to store the descriptor vector of
            floats in.
        :param timestamp_field: Solr index field to store floating-point UNIX
            timestamps.
        :param timeout: Whether or not the Solr connection should
            be persistent or not.
        :param persistent_connection: Maintain a connection between Solr index
            interactions.
        :param commit_on_set: Immediately commit changes when a vector is set.
        """
        super(SolrDescriptorElement, self).__init__(uuid)

        self.uuid_field = uuid_field
        self.vector_field = vector_field
        self.timestamp_field = timestamp_field

        self.solr_conn_addr = solr_conn_addr
        self.solr_timeout = timeout
        self.solr_persistent_connection = persistent_connection
        self.solr_commit_on_set = commit_on_set

        self.solr = self._make_solr_inst()

    def __getstate__(self) -> Dict[str, Any]:
        state = super(SolrDescriptorElement, self).__getstate__()
        state.update({
            "uuid_field": self.uuid_field,
            "vector_field": self.vector_field,
            "timestamp_field": self.timestamp_field,
            "solr_conn_addr": self.solr_conn_addr,
            "solr_persistent_connection": self.solr_persistent_connection,
            "solr_timeout": self.solr_timeout,
            "solr_commit_on_set": self.solr_commit_on_set,
        })
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.uuid_field = state['uuid_field']
        self.vector_field = state['vector_field']
        self.timestamp_field = state['timestamp_field']
        self.solr_conn_addr = state['solr_conn_addr']
        self.solr_timeout = state['solr_timeout']
        self.solr_persistent_connection = state['solr_persistent_connection']
        self.solr_commit_on_set = state['solr_commit_on_set']

        self.solr = self._make_solr_inst()

    def __repr__(self) -> str:
        return super(SolrDescriptorElement, self).__repr__() + \
            '[url: %s, timeout: %d, ' \
            'persistent: %s]' \
            % (self.solr.url, self.solr.timeout, self.solr.persistent)

    def _make_solr_inst(self) -> "solr.Solr":
        return solr.Solr(self.solr_conn_addr,
                         persistent=self.solr_persistent_connection,
                         timeout=self.solr_timeout,
                         # debug=True  # This makes things pretty verbose
                         )

    def _base_doc(self) -> Dict[str, Any]:
        """
        :returns: A new dictionary representing the basic document structure
            for interacting with our elements in Solr.
        """
        suuid = str(self.uuid())
        return {
            'id': '-'.join([suuid]),
            self.uuid_field: suuid,
        }

    def _get_existing_doc(self) -> Optional[Dict[str, Any]]:
        """
        :return: An existing document dict. If there isn't one for our uuid
            we return None.
        """
        b_doc = self._base_doc()
        r = self.solr.select(f"id:{b_doc['id']} \
                             AND {self.uuid_field}:{b_doc[self.uuid_field]}")
        if r.numFound == 1:
            return r.results[0]
        else:
            return None

    def get_config(self) -> Dict[str, Any]:
        return {
            "solr_conn_addr": self.solr_conn_addr,
            "uuid_field": self.uuid_field,
            "vector_field": self.vector_field,
            "timestamp_field": self.timestamp_field,
            "timeout": self.solr_timeout,
            "persistent_connection": self.solr_persistent_connection,
            "commit_on_set": self.solr_commit_on_set,
        }

    def has_vector(self) -> bool:
        return bool(self._get_existing_doc())

    def set_vector(self, new_vec: numpy.ndarray) -> "SolrDescriptorElement":
        doc = self._base_doc()
        doc[self.vector_field] = new_vec.tolist()
        doc[self.timestamp_field] = time.time()
        self.solr.add(doc, commit=self.solr_commit_on_set)
        return self

    def vector(self) -> Optional[numpy.ndarray]:
        doc = self._get_existing_doc()
        if doc is None:
            return None
        # Vectors stored as lists in solr doc
        return numpy.array(doc[self.vector_field])
