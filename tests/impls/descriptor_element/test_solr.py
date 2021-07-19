import unittest

import unittest.mock as mock
import pytest

from smqtk_core.configuration import configuration_test_helper
from smqtk_descriptors.impls.descriptor_element.solr import SolrDescriptorElement


@pytest.mark.skipif(not SolrDescriptorElement.is_usable(),
                    reason='SolrDescriptorElement reports as not usable.')
class TestDescriptorSolrElement (unittest.TestCase):

    TEST_URL = 'http://localhost:8983/solr'

    @mock.patch("solr.Solr")
    def test_configuration(self, _mock_solr: mock.MagicMock) -> None:
        inst = SolrDescriptorElement(
            'a',
            solr_conn_addr=self.TEST_URL,
            uuid_field='uuid_s', vector_field='vector_fs',
            timestamp_field='timestamp_f', timeout=101,
            persistent_connection=True, commit_on_set=False,
        )
        for i in configuration_test_helper(inst, {'uuid'}, ('abcd',)):
            assert i.solr_conn_addr == self.TEST_URL
            assert i.uuid_field == 'uuid_s'
            assert i.vector_field == 'vector_fs'
            assert i.timestamp_field == 'timestamp_f'
            assert i.solr_timeout == 101
            assert i.solr_persistent_connection is True
            assert i.solr_commit_on_set is False
