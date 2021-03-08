"""
References:
    [1] https://stackoverflow.com/a/29598910
"""
import logging
import multiprocessing
import pickle
from typing import Any, Dict, Generator, Hashable, Iterable, Optional, Sequence, Tuple

from smqtk_dataprovider.exceptions import ReadOnlyError
from smqtk_dataprovider.utils.postgres import norm_psql_cmd_string, PsqlConnectionHelper
from smqtk_descriptors import DescriptorElement, DescriptorSet


LOG = logging.getLogger(__name__)


try:
    import psycopg2  # type: ignore
    import psycopg2.extensions  # type: ignore
except ImportError as ex:
    LOG.warning("Failed to import psycopg2: %s", str(ex))
    psycopg2 = None


PSQL_TABLE_CREATE_RLOCK = multiprocessing.RLock()


# noinspection SqlNoDataSourceInspection
class PostgresDescriptorSet (DescriptorSet):
    """
    DescriptorSet implementation that stored DescriptorElement references in
    a PostgreSQL database.

    A ``PostgresDescriptorSet`` effectively controls the entire table. Thus
    a ``clear()`` call will remove everything from the table.

    PostgreSQL version support:
        - 9.4

    Table format:
        <uuid col>      TEXT NOT NULL
        <element col>   BYTEA NOT NULL

        <uuid_col> should be the primary key (we assume unique).

    We require that the no column labels not be 'true' for the use of a value
    return shortcut.

    """

    #
    # The following are SQL query templates. The string formatting using {}'s
    # is used to fill in the query before using it in an execute with instance
    # specific values. The ``%()s`` formatting is special for the execute
    # where-by psycopg2 will fill in the values appropriately as specified in a
    # second dictionary argument to ``cursor.execute(query, value_dict)``.
    #
    UPSERT_TABLE_TMPL = norm_psql_cmd_string("""
        CREATE TABLE IF NOT EXISTS {table_name:s} (
          {uuid_col:s} TEXT NOT NULL,
          {element_col:s} BYTEA NOT NULL,
          PRIMARY KEY ({uuid_col:s})
        );
    """)

    SELECT_TMPL = norm_psql_cmd_string("""
        SELECT {col:s}
          FROM {table_name:s}
    """)

    SELECT_LIKE_TMPL = norm_psql_cmd_string("""
        SELECT {element_col:s}
          FROM {table_name:s}
         WHERE {uuid_col:s} like %(uuid_like)s
    """)

    # So we can ensure we get back elements in specified order
    #   - reference [1]
    SELECT_MANY_ORDERED_TMPL = norm_psql_cmd_string("""
        SELECT {table_name:s}.{element_col:s}
          FROM {table_name:s}
          JOIN (
            SELECT *
            FROM unnest(%(uuid_list)s) with ordinality
          ) AS __ordering__ ({uuid_col:s}, {uuid_col:s}_order)
            ON {table_name:s}.{uuid_col:s} = __ordering__.{uuid_col:s}
          ORDER BY __ordering__.{uuid_col:s}_order
    """)

    UPSERT_TMPL = norm_psql_cmd_string("""
        WITH upsert AS (
          UPDATE {table_name:s}
            SET {element_col:s} = %(element_val)s
            WHERE {uuid_col:s} = %(uuid_val)s
            RETURNING *
          )
        INSERT INTO {table_name:s}
          ({uuid_col:s}, {element_col:s})
          SELECT %(uuid_val)s, %(element_val)s
            WHERE NOT EXISTS (SELECT * FROM upsert)
    """)

    DELETE_LIKE_TMPL = norm_psql_cmd_string("""
        DELETE FROM {table_name:s}
              WHERE {uuid_col:s} like %(uuid_like)s
    """)

    DELETE_MANY_TMPL = norm_psql_cmd_string("""
        DELETE FROM {table_name:s}
              WHERE {uuid_col:s} in %(uuid_tuple)s
          RETURNING uid
    """)

    @classmethod
    def is_usable(cls) -> bool:
        return psycopg2 is not None

    def __init__(
        self,
        table_name: str = 'descriptor_set',
        uuid_col: str = 'uid',
        element_col: str = 'element',
        db_name: str = 'postgres',
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_user: Optional[str] = None,
        db_pass: Optional[str] = None,
        multiquery_batch_size: Optional[int] = 1000,
        pickle_protocol: int = -1,
        read_only: bool = False,
        create_table: bool = True
    ):
        """
        Initialize set instance.

        :param table_name: Name of the table to use.
        :param uuid_col: Name of the column containing the UUID signatures.
        :param element_col: Name of the table column that will contain
            serialized elements.
        :param db_name: The name of the database to connect to.
        :param db_host: Host address of the Postgres server. If None, we
            assume the server is on the local machine and use the UNIX socket.
            This might be a required field on Windows machines (not tested yet).
        :param db_port: Port the Postgres server is exposed on. If None, we
            assume the default port (5423).
        :param db_user: Postgres user to connect as. If None, postgres
            defaults to using the current accessing user account name on the
            operating system.
        :param db_pass: Password for the user we're connecting as. This may be
            None if no password is to be used.
        :param multiquery_batch_size: For queries that handle sending or
            receiving many queries at a time, batch queries based on this size.
            If this is None, then no batching occurs.

            The advantage of batching is that it reduces the memory impact for
            queries dealing with a very large number of elements (don't have to
            store the full query for all elements in RAM), but the transaction
            will be some amount slower due to splitting the query into multiple
            transactions.
        :param pickle_protocol: Pickling protocol to use. We will use -1 by
            default (latest version, probably binary).
        :param read_only: Only allow read actions against this set.
            Modification actions will throw a ReadOnlyError exceptions.
        :param create_table: If this instance should try to create the storing
            table before actions are performed against it when not set to be
            read-only. If the configured user does not have sufficient
            permissions to create the table and it does not currently exist, an
            exception will be raised.
        """
        super(PostgresDescriptorSet, self).__init__()

        self.table_name = table_name
        self.uuid_col = uuid_col
        self.element_col = element_col

        self.multiquery_batch_size = multiquery_batch_size
        self.pickle_protocol = pickle_protocol
        self.read_only = bool(read_only)
        self.create_table = create_table

        # Checking parameters where necessary
        if self.multiquery_batch_size is not None:
            self.multiquery_batch_size = int(self.multiquery_batch_size)
            assert self.multiquery_batch_size > 0, \
                "A given batch size must be greater than 0 in size " \
                "(given: %d)." % self.multiquery_batch_size
        assert -1 <= self.pickle_protocol <= 2, \
            ("Given pickle protocol is not in the known valid range. Given: %s"
             % self.pickle_protocol)

        self.psql_helper = PsqlConnectionHelper(db_name, db_host, db_port,
                                                db_user, db_pass,
                                                self.multiquery_batch_size,
                                                PSQL_TABLE_CREATE_RLOCK)
        if not self.read_only and self.create_table:
            self.psql_helper.set_table_upsert_sql(
                self.UPSERT_TABLE_TMPL.format(
                    table_name=self.table_name,
                    uuid_col=self.uuid_col,
                    element_col=self.element_col,
                )
            )

    def get_config(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "uuid_col": self.uuid_col,
            "element_col": self.element_col,

            "db_name": self.psql_helper.db_name,
            "db_host": self.psql_helper.db_host,
            "db_port": self.psql_helper.db_port,
            "db_user": self.psql_helper.db_user,
            "db_pass": self.psql_helper.db_pass,

            "multiquery_batch_size": self.multiquery_batch_size,
            "pickle_protocol": self.pickle_protocol,
            "read_only": self.read_only,
            "create_table": self.create_table,
        }

    def count(self) -> int:
        """
        :return: Number of descriptor elements stored in this set.
        """
        # Just count UUID column to limit data read.
        q = self.SELECT_TMPL.format(
            col='count(%s)' % self.uuid_col,
            table_name=self.table_name,
        )

        def exec_hook(cur: psycopg2.extensions.cursor) -> None:
            cur.execute(q)

        # There's only going to be one row returned with one element in it.
        return list(self.psql_helper.single_execute(
            exec_hook, yield_result_rows=True
        ))[0][0]

    def clear(self) -> None:
        if self.read_only:
            raise ReadOnlyError("Cannot clear a read-only set.")

        q = self.DELETE_LIKE_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )

        def exec_hook(cur: psycopg2.extensions.cursor) -> None:
            cur.execute(q, {'uuid_like': '%'})

        list(self.psql_helper.single_execute(exec_hook))

    def has_descriptor(self, uuid: Hashable) -> bool:
        q = self.SELECT_LIKE_TMPL.format(
            # hacking return value to something simple
            element_col='true',
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )

        def exec_hook(cur: psycopg2.extensions.cursor) -> None:
            cur.execute(q, {'uuid_like': str(uuid)})

        # Should either yield one or zero rows
        return bool(list(self.psql_helper.single_execute(
            exec_hook, yield_result_rows=True
        )))

    def add_descriptor(self, descriptor: DescriptorElement) -> None:
        """
        Add a descriptor to this set.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the set (based on UUID). Added descriptors
        overwrite set descriptors based on UUID.

        :param descriptor: Descriptor to set.
        """
        if self.read_only:
            raise ReadOnlyError("Cannot clear a read-only set.")

        q = self.UPSERT_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
            element_col=self.element_col,
        )
        v = {
            'uuid_val': str(descriptor.uuid()),
            'element_val': psycopg2.Binary(
                pickle.dumps(descriptor, self.pickle_protocol)
            )
        }

        def exec_hook(cur: psycopg2.extensions.cursor) -> None:
            cur.execute(q, v)

        list(self.psql_helper.single_execute(exec_hook))

    def add_many_descriptors(self, descriptors: Iterable[DescriptorElement]) -> None:
        """
        Add multiple descriptors at one time.

        Adding the same descriptor multiple times should not add multiple copies
        of the descriptor in the set (based on UUID). Added descriptors
        overwrite set descriptors based on UUID.

        :param descriptors: Iterable of descriptor instances to add to this
            set.
        """
        if self.read_only:
            raise ReadOnlyError("Cannot clear a read-only set.")

        q = self.UPSERT_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
            element_col=self.element_col,
        )

        # Transform input into
        def iter_elements() -> Generator[Dict[str, Any], None, None]:
            for d in descriptors:
                yield {
                    'uuid_val': str(d.uuid()),
                    'element_val': psycopg2.Binary(
                        pickle.dumps(d, self.pickle_protocol)
                    )
                }

        def exec_hook(cur: psycopg2.extensions.cursor, batch: Sequence[Dict[str, Any]]) -> None:
            cur.executemany(q, batch)

        LOG.debug("Adding many descriptors")
        list(self.psql_helper.batch_execute(iter_elements(), exec_hook,
                                            self.multiquery_batch_size))

    def get_descriptor(self, uuid: Hashable) -> DescriptorElement:
        """
        Get the descriptor in this set that is associated with the given UUID.

        :param uuid: UUID of the DescriptorElement to get.

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this set.

        :return: DescriptorElement associated with the queried UUID.
        """
        q = self.SELECT_LIKE_TMPL.format(
            element_col=self.element_col,
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )
        v = {'uuid_like': str(uuid)}

        def eh(c: psycopg2.extensions.cursor) -> None:
            c.execute(q, v)
            if c.rowcount == 0:
                raise KeyError(uuid)
            elif c.rowcount != 1:
                raise RuntimeError("Found more than one entry for the given "
                                   "uuid '%s' (got: %d)"
                                   % (uuid, c.rowcount))

        r = list(self.psql_helper.single_execute(eh, yield_result_rows=True))
        return pickle.loads(bytes(r[0][0]))

    def get_many_descriptors(self, uuids: Iterable[Hashable]) -> Generator[DescriptorElement, None, None]:
        """
        Get an iterator over descriptors associated to given descriptor UUIDs.

        :param uuids: Iterable of descriptor UUIDs to query for.

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this set.

        :return: Iterator of descriptors associated to given uuid values.
        """
        q = self.SELECT_MANY_ORDERED_TMPL.format(
            table_name=self.table_name,
            element_col=self.element_col,
            uuid_col=self.uuid_col,
        )

        # Cache UUIDs received in order so we can check when we miss one in
        # order to raise a KeyError.
        uuid_order = []

        def iterelems() -> Generator[str, None, None]:
            for uid in uuids:
                uuid_order.append(uid)
                yield str(uid)

        def exec_hook(cur: psycopg2.extensions.cursor, batch: Sequence[str]) -> None:
            v = {'uuid_list': batch}
            # LOG.debug('query: %s', cur.mogrify(q, v))
            cur.execute(q, v)

        LOG.debug("Getting many descriptors")
        # The SELECT_MANY_ORDERED_TMPL query ensures that elements returned are
        #   in the UUID order given to this method. Thus, if the iterated UUIDs
        #   and iterated return rows do not exactly line up, the query join
        #   failed to match a query UUID to something in the database.
        #   - We also check that the number of rows we got back is the same
        #     as elements yielded, else there were trailing UUIDs that did not
        #     match anything in the database.
        g = self.psql_helper.batch_execute(iterelems(), exec_hook,
                                           self.multiquery_batch_size,
                                           yield_result_rows=True)
        i = 0
        for r, expected_uuid in zip(g, uuid_order):
            d = pickle.loads(bytes(r[0]))
            if d.uuid() != expected_uuid:
                raise KeyError(expected_uuid)
            yield d
            i += 1

        if len(uuid_order) != i:
            # just report the first one that's bad
            raise KeyError(uuid_order[i])

    def remove_descriptor(self, uuid: Hashable) -> None:
        """
        Remove a descriptor from this set by the given UUID.

        :param uuid: UUID of the DescriptorElement to remove.

        :raises KeyError: The given UUID doesn't associate to a
            DescriptorElement in this set.
        """
        if self.read_only:
            raise ReadOnlyError("Cannot remove from a read-only set.")

        q = self.DELETE_LIKE_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )
        v = {'uuid_like': str(uuid)}

        def execute(c: psycopg2.extensions.cursor) -> None:
            c.execute(q, v)
            # Nothing deleted if rowcount == 0
            # (otherwise 1 when deleted a thing)
            if c.rowcount == 0:
                raise KeyError(uuid)

        list(self.psql_helper.single_execute(execute))

    def remove_many_descriptors(self, uuids: Iterable[Hashable]) -> None:
        """
        Remove descriptors associated to given descriptor UUIDs from this set.

        :param uuids: Iterable of descriptor UUIDs to remove.

        :raises KeyError: A given UUID doesn't associate with a
            DescriptorElement in this set.
        """
        if self.read_only:
            raise ReadOnlyError("Cannot remove from a read-only set.")

        q = self.DELETE_MANY_TMPL.format(
            table_name=self.table_name,
            uuid_col=self.uuid_col,
        )
        str_uuid_set = set(str(uid) for uid in uuids)
        v = {'uuid_tuple': tuple(str_uuid_set)}

        def execute(c: psycopg2.extensions.cursor) -> None:
            c.execute(q, v)

            # Check query UUIDs against rows that would actually be deleted.
            deleted_uuid_set = set(r[0] for r in c.fetchall())
            for uid in str_uuid_set:
                if uid not in deleted_uuid_set:
                    raise KeyError(uid)

        list(self.psql_helper.single_execute(execute))

    def iterkeys(self) -> Generator[Hashable, None, None]:
        """
        Return an iterator over set descriptor keys, which are their UUIDs.
        """
        # Getting UUID through the element because the UUID might not be a
        # string type, and the true type is encoded with the DescriptorElement
        # instance.
        for d in self.iterdescriptors():
            yield d.uuid()

    def iterdescriptors(self) -> Generator[DescriptorElement, None, None]:
        """
        Return an iterator over set descriptor element instances.
        """
        def execute(c: psycopg2.extensions.cursor) -> None:
            c.execute(self.SELECT_TMPL.format(
                col=self.element_col,
                table_name=self.table_name
            ))

        #: :type: __generator
        execution_results = self.psql_helper.single_execute(
            execute, yield_result_rows=True, named=True
        )
        for r in execution_results:
            d = pickle.loads(bytes(r[0]))
            yield d

    def iteritems(self) -> Generator[Tuple[Hashable, DescriptorElement], None, None]:
        """
        Return an iterator over set descriptor key and instance pairs.
        :rtype: collections.abc.Iterator[(collections.abc.Hashable,
                                          smqtk.representation.DescriptorElement)]
        """
        for d in self.iterdescriptors():
            yield d.uuid(), d
