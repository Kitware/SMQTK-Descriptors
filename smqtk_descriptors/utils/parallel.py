import abc
from collections.abc import Iterator as abc_Iterator
from itertools import zip_longest
import heapq
import logging
import multiprocessing
import multiprocessing.queues
import multiprocessing.synchronize
import queue
import sys
import threading
import traceback
from typing import Any, Callable, Iterable, List, Optional, Sequence, Type, Union


LOG = logging.getLogger(__name__)


def parallel_map(
    work_func: Callable,
    *sequences: Iterable,
    **kwargs: Any
) -> "ParallelResultsIterator":
    """
    Generalized local parallelization helper for executing embarrassingly
    parallel functions on an iterable of input data. This function then yields
    work results for data, optionally in the order that they were provided to
    this function.

    By default, we act like ``itertools.izip`` in regards to input sequences,
    whereby we stop performing work as soon as one of the input sequences is
    exhausted. The optional keyword argument ``fill_void`` may be specified to
    enable sequence handling like ``itertools.zip_longest`` where the longest
    sequence determines what is iterated, and the value given to ``fill_void``
    is used as the fill value.

    This is intended to be able to replace ``multiprocessing.pool.Pool`` and
    ``multiprocessing.pool.ThreadPool`` uses with the added benefit of:

        - No set-up or clean-up needed
        - No performance loss compared to ``multiprocessing.pool`` classes
          for non-trivial work functions (like IO operations).
        - We can iterate results as they are ready (optionally in order of
          input)
        - Lambda or on-the-fly function can be provided as the work function
          when using multiprocessing.
        - Buffered input/output queues so that mapping work of a very large
          input set doesn't overrun your memory (e.g. iterating over many large
          vectors/matrices).

    This function is, however, slower than multiprocessing.pool classes for
    trivial functions, like using the function ``ord`` over a set of
    characters.

    Input data given to ``sequences`` must be picklable in order to transport
    to worker threads/processes.

    Input Iteration and Results Buffering
    -------------------------------------
    The buffer factor, ``F``, operates with the number of utilized cores,
    ``C``, to create an upper bound on the times the input sequences are
    iterated and the number of work function outputs are held in memory at any
    given time.

    The maximum number of input sequence items loaded at a time is
    ``floor(C * F) + C``.
    This is due to the input work queue ``maxsize`` being set to ``floor(C*F)``
    while there can be ``C`` workers could be utilizing their inputs to
    complete their work instances.

    The maximum number of results queued is ``floor(C * F) + C``.
    This is similarly due to the output result queue maxsize being set to
    ``floor(C * F)`` while there can be ``C`` workers blocked on putting values
    into a full results queue.

    Sometimes its important to know how much farther ahead the input
    iterator(s) have yielded compared to the number of output results from the
    ``ParallelResultsIterator``.
    For some yielded result at index ``N``, the input iterator(s) next yielded
    item should be their index ``N + (2 * floor(C * F) + C)``. This is
    derived from the input work and output result queues maximally filled with
    at most ``floor(C * F)`` items and there being ``C`` workers working on, or
    attempting to queue results for, their current inputs.
    For example, if we have use ``C=4`` and ``F=1.5``, if result index N has just
    been yielded, then the input iterators are ready to yield their
    ``N + 16``-th indexed item (``2 * floor(4*1.5) + 4 = 2 * 6 + 4 = 16``).

    The above is only guaranteed no the ``ordered`` option is ``False``,
    otherwise non-determinism in processing order can cause results for input
    items to return out of order, causing additional buffering in the heap used
    to ensure ordered output which, necessarily, has no size limits so as to
    not dead-lock.

    :param work_func:
        Function that performs some work on input data, resulting in some
        returned value.

        When in multiprocessing mode, this cannot be a local function or a
        transport error will occur when trying to move the function to the
        worker process.
    :param sequences: Input data to apply to the given ``work_func`` function.
        If more than one sequence is given, the function is called with an
        argument list consisting of the corresponding item of each sequence.
        While we expect Iterable types to be provided here, they will only be
        observed by `zip`/`zip_longest` at most once. See iteration rules for
        `zip`/`zip_longest` for details.
    :param kwargs: Optionally available keyword arguments are as follows:

        - fill_void
            - Optional value that, if specified, activates sequence handling
              like that of ``itertools.zip_longest``, using the provided value
              as a fill-in for shorter sequences until the longest sequence is
              exhausted.
            - type: Any
            - default: No default

        - ordered
            - If results for input elements should be yielded in the same order
              as input elements. If False, we yield results as soon as they are
              collected.
            - type: bool
            - default: True

        - buffer_factor
            - Multiplier against the number of processes used to limit the
              growth size of the result queue coming from worker processes
              (``int(cores * buffer_factor)``). This is utilized so we don't
              overrun our RAM buffering results.
            - type: float
            - default: 2.0

        - cores
            - Optional specification of the number of threads/cores to use. If
              None, we will attempt to use all available threads/cores.
            - type: None | int | long
            - default: None

        - use_multiprocessing
            - Whether or not to use discrete processes as the parallelization
              agent vs python threads.
            - type: bool
            - default: False

        - heart_beat
            - Interval at which workers check for operational messages while
              waiting on locks (e.g. waiting to push or pull messages). This
              ensures that workers are not left hanging, or hang the program,
              when and error or interruption occurs, or when waiting on an full
              edge. This must be >0.
            - type: float
            - default: 0.001

        - name
            - Optional string name for identifying workers and logging
              messages. ``None`` means no names are added.
            - type: str
            - default: None

        - daemon
            - Optional flag for if started threads/processes are flagged as
              daemonic. This is should probably nearly always be True (the
              default) otherwise related threads/processes can hang the main
              process.
            - type: bool
            - default: True

    :return: A new parallel results iterator that starts work on the input
        iterable when iterated.


    Example
    -------
    >>> import math
    >>> result_iter = parallel_map(math.factorial, range(10),
    ...                            use_multiprocessing=True)
    >>> sorted(result_iter)
    [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
    """
    # kwargs
    cores: Optional[int] = kwargs.get('cores', None)
    ordered = kwargs.get('ordered', True)
    buffer_factor = kwargs.get('buffer_factor', 2.0)
    use_multiprocessing = kwargs.get('use_multiprocessing', False)
    heart_beat = kwargs.get('heart_beat', 0.001)
    fill_activate = 'fill_void' in kwargs
    fill_value = kwargs.get('fill_void', None)
    name = kwargs.get('name', None)
    daemon = kwargs.get('daemon', True)

    if name:
        log = logging.getLogger(__name__ + '[%s]' % name)
    else:
        log = logging.getLogger(__name__)

    if heart_beat <= 0:
        raise ValueError("heart_beat must be >0.")

    if cores is None or cores <= 0:
        cores = multiprocessing.cpu_count()
        log.debug("Using all cores (%d)", cores)
    else:
        log.debug("Only using %d cores", cores)

    # Choose parallel types
    queue_t: Union[Type[multiprocessing.Queue], Type[queue.Queue]]
    worker_t: Type[Union[_WorkerThread, _WorkerProcess]]
    if use_multiprocessing:
        queue_t = multiprocessing.Queue
        worker_t = _WorkerProcess
    else:
        queue_t = queue.Queue
        worker_t = _WorkerThread

    # Type ignoring these calls due to a mypy issue where it's deducing the
    # type to `<nothing>` for some reason.
    queue_work = queue_t(maxsize=int(cores * buffer_factor))  # type: ignore
    queue_results = queue_t(maxsize=int(cores * buffer_factor))  # type: ignore

    log.log(1, "Constructing worker processes")
    workers = [worker_t(name, i, work_func, queue_work, queue_results,
                        heart_beat)
               for i in range(cores)]

    log.log(1, "Constructing feeder thread")
    feeder_thread = _FeedQueueThread(name, sequences, queue_work,
                                     len(workers), heart_beat, fill_activate,
                                     fill_value)

    return ParallelResultsIterator(name, ordered, use_multiprocessing,
                                   heart_beat, queue_work,
                                   queue_results, feeder_thread, workers,
                                   daemon)


class _TerminalPacket (object):
    """
    Signals a terminal message
    """


def _is_terminal(p: Any) -> bool:
    """
    Check if a given packet is a terminal element.

    :param p: element to check

    :return: If ``p`` is a terminal element
    """
    return isinstance(p, _TerminalPacket)


class ParallelResultsIterator (abc_Iterator):
    """
    Iterator return from a parallel mapping job, managing workers and output
    results queue consumption.

    A parallel work mapping jobs may be canceled through this object.

    :param name: String name to attribute to this iterator. May be None.
    :param ordered: If this results iterator should yield results in a
        congruent order to the input parameter sequences. If this is
        `false` then this results iterator will yield results as soon as
        they are available regardless of the input parameter sequence
        order.
    :param is_multiprocessing: If workers are processes vs. threads. When
        this is true, extra steps are taken to appropriately shutdown
        processes.
    :param heart_beat: How long in seconds we wait when polling for data on
        the results queue before momentarily giving up to allow a cycle of
        the loop. This is important in allowing an external signal to
        indicate we should stop iterating (prevents hanging on getting the
        next result value).
    :param work_queue: Queue into which work is placed by the feeder
        thread. This object is responsible for cleaning up this queue, if
        applicable, upon iteration termination.
    :param results_queue: Queue from which work results are pulled. This
        object is responsible for cleaning up this queue, if applicable,
        upon iteration termination.
    :param feeder_thread: Thread for feeding the input queue for this
        iterator to manage starting and stopping appropriately.
    :param workers: Sequence of worker threads/processes for this iterator
        to manage starting and stopping appropriately.
    :param daemon: If the managed threads/processes should be started as
        daemons.
    """

    def __init__(
        self,
        name: Optional[str],
        ordered: bool,
        is_multiprocessing: bool,
        heart_beat: float,
        work_queue: Union[queue.Queue, multiprocessing.Queue],
        results_queue: Union[queue.Queue, multiprocessing.Queue],
        feeder_thread: "_FeedQueueThread",
        workers: Sequence[Union["_WorkerThread", "_WorkerProcess"]],
        daemon: bool
    ):
        self.name = name
        self._l_prefix: str = f"[PRI{(name and f'::{name}') or ''}]"

        self.ordered = ordered
        if self.ordered:
            LOG.debug(f"{self._l_prefix} Maintaining result iteration order "
                      f"based on input order")
        self.heart_beat = heart_beat
        self.is_multiprocessing = is_multiprocessing

        self.work_queue = work_queue
        self.results_queue = results_queue
        self.feeder_thread = feeder_thread
        self.workers = workers
        self.daemon = daemon

        self.has_started_workers = False
        self.has_cleaned_up = False

        self.found_terminals = 0
        self.result_heap: List = []
        self.next_index = 0

        self.stop_event = threading.Event()
        self.stop_event_lock = threading.Lock()

    def __repr__(self) -> str:
        sfx = ''
        if self.name:
            sfx = '[' + self.name + ']'
        return "<%(module)s.%(class)s%(sfx)s at %(address)s>" % {
            "module": self.__module__,
            "class": self.__class__.__name__,
            "sfx": sfx,
            "address": hex(id(self)),
        }

    def __next__(self) -> Any:
        l_prefix = self._l_prefix
        try:
            if not self.has_started_workers:
                self.start_workers()

            while (self.found_terminals < len(self.workers) and
                   not self.stopped()):
                packet = self.results_q_get()

                if _is_terminal(packet):
                    LOG.log(1, f'{l_prefix} Found terminal')
                    self.found_terminals += 1
                elif isinstance(packet[0], BaseException):
                    ex, formatted_exc = packet
                    LOG.warning(f'{l_prefix} Received exception: '
                                f'{ex}\n{formatted_exc}')
                    raise ex
                else:
                    i, result = packet
                    if self.ordered:
                        heapq.heappush(self.result_heap, (i, result))
                        if self.result_heap[0][0] == self.next_index:
                            _, result = heapq.heappop(self.result_heap)
                            self.next_index += 1
                            return result
                    else:
                        return result

            # Go through heap if there's anything in it
            if self.result_heap:
                _, result = heapq.heappop(self.result_heap)
                return result

            # Nothing left
            if not self.stopped():
                LOG.log(1, f"{l_prefix} Asserting empty queues on what looks "
                           f"like a full iteration.")
                self.assert_queues_empty()

            raise StopIteration()

        # If anything bad happens, stop iteration and workers.
        # - Using BaseException to also catch things like KeyboardInterrupt
        #   and other exceptions that do not descend from Exception.
        # - This also catches the in-due-course StopIteration exception, thus
        #   this is also the "normal" stop route.
        except BaseException as ex:
            LOG.log(1, f"{l_prefix} Stopping iteration due to exception: "
                       f"({type(ex)}) {str(ex)}")
            self.stop()
            raise

    next = __next__

    def start_workers(self) -> None:
        """
        Start worker threads/processes.
        """
        LOG.log(1, f"{self._l_prefix} Starting worker processes")
        for w in self.workers:
            w.daemon = self.daemon
            w.start()

        LOG.log(1, f"{self._l_prefix} Starting feeder thread")
        self.feeder_thread.daemon = self.daemon
        self.feeder_thread.start()

        self.has_started_workers = True

    def clean_up(self) -> None:
        """
        Clean up any live resources if we haven't done so already.
        """
        if self.has_started_workers and not self.has_cleaned_up:
            l_prefix = self._l_prefix

            LOG.log(1, f"{l_prefix} Stopping feeder thread")
            self.feeder_thread.stop()
            self.feeder_thread.join()

            LOG.log(1, f"{l_prefix} Stopping workers")
            for w in self.workers:
                w.stop()
                w.join()

            if self.is_multiprocessing:
                LOG.log(1, f"{l_prefix} Closing/Joining process queues")
                for q in (self.work_queue, self.results_queue):
                    assert isinstance(q, multiprocessing.queues.Queue)
                    q.close()
                    q.join_thread()

            self.has_cleaned_up = True

    def stop(self) -> None:
        """
        Stop this iterator.

        This does not clean up resources (see ``clean_up`` for that).
        """
        with self.stop_event_lock:
            self.stop_event.set()
            self.clean_up()

    def stopped(self) -> bool:
        """
        :return: if this iterator has been stopped
        """
        return self.stop_event.is_set()

    def results_q_get(self) -> Any:
        """
        Attempts to get something from the results queue.

        :raises StopIteration: when we've been told to stop.
        :returns: Single result from the results queue.
        """
        while not self.stopped():
            try:
                return self.results_queue.get(timeout=self.heart_beat)
            except queue.Empty:
                pass
        raise StopIteration()

    def assert_queues_empty(self) -> None:
        # All work should be exhausted at this point
        if self.is_multiprocessing and sys.platform == 'darwin':
            # multiprocessing.Queue.qsize doesn't work on OSX
            # - Try to get something from each queue, expecting an empty
            #   exception.
            # - multiprocessing shares the same exception as the queue module.
            try:
                self.work_queue.get(block=False)
            except queue.Empty:
                pass
            else:
                raise AssertionError("In queue not empty")
            try:
                self.results_queue.get(block=False)
            except queue.Empty:
                pass
            else:
                raise AssertionError("Out queue not empty")
        else:
            assert self.work_queue.qsize() == 0, \
                "In queue not empty (%d)" % self.work_queue.qsize()
            assert self.results_queue.qsize() == 0, \
                "Out queue not empty (%d)" % self.results_queue.qsize()


class _FeedQueueThread (threading.Thread):
    """
    Helper thread for putting data into the work queue

    """

    def __init__(
        self,
        name: Optional[str],
        arg_sequences: Sequence[Iterable],
        q: Union[queue.Queue, multiprocessing.Queue],
        num_terminal_packets: int,
        heart_beat: float,
        do_fill: bool,
        fill_value: Any
    ):
        """
        :param name: Optional name for this feed queue thread.
        :param arg_sequences: Sequence of iterators that will feed input work
            arguments. While we expect Iterable types to be provided here, they
            will only be observed by `zip`/`zip_longest` at most once. See
            iteration rules for `zip`/`zip_longest` for details.
        :param q: Queue to put work argument sets into.
        :param num_terminal_packets: Number of terminal packets to put into the
            work queue upon completion of submitting real work. This should be
            the same number of workers feeding off of `q`.
        :param heart_beat: How long in seconds we wait for an individual put
            attempt into `q` before momentarily giving up to allow a cycle of
            the loop. This is important in allowing an external signal to
            indicate we should stop feeding efforts (prevents hanging on
            pushing values into `q`).
        :param do_fill: If we should fill in a certain value for the shorter
            input sequences along the same rules for `itertools.zip_longest`.
        :param fill_value: The value to fill with if `do_fill` is True.
        """
        super().__init__(name=name)
        self._l_prefix: str = f"[FQT{(name and f'::{name}') or ''}]"

        self.arg_sequences = arg_sequences
        self.q = q
        self.num_terminal_packets = num_terminal_packets
        self.heart_beat = heart_beat
        self.do_fill = do_fill
        self.fill_value = fill_value

        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        l_prefix = self._l_prefix
        LOG.log(1, f"{l_prefix} Starting")

        if self.do_fill:
            _zip = zip_longest
            _zip_kwds = {'fillvalue': self.fill_value}
        else:
            # Ignoring that the callable here doesn't *exactly* match
            # zip_longest based on stubs: They match in the way that we use it
            # here.
            _zip = zip  # type: ignore
            _zip_kwds = {}

        try:
            r = 0
            for args in _zip(*self.arg_sequences, **_zip_kwds):
                self.q_put((r, args))
                r += 1

                # If we're told to stop, immediately quit out of processing
                if self.stopped():
                    LOG.log(1, f"{l_prefix} Told to stop prematurely")
                    break
        # Transport back any exceptions raised
        except (Exception, KeyboardInterrupt) as ex:
            LOG.warning(f"{l_prefix} Caught exception {str(ex)}")
            self.q_put((ex, traceback.format_exc()))
            self.stop()
        else:
            LOG.log(1, f"{l_prefix} Sending in-queue terminal packets")
            for _ in range(self.num_terminal_packets):
                self.q_put(_TerminalPacket())
        finally:
            # Explicitly stop any nested parallel maps
            for s in self.arg_sequences:
                if isinstance(s, ParallelResultsIterator):
                    LOG.log(1, f"{l_prefix} Stopping nested parallel map: {s}")
                    s.stop()

            LOG.log(1, f"{l_prefix} Closing")

    def q_put(self, val: Any) -> None:
        """
        Try to put the given value into the output queue until it is inserted
        (if it was previously full), or the stop signal was given.

        :param val: value to put into the output queue.
        """
        put = False
        while not put and not self.stopped():
            try:
                self.q.put(val, timeout=self.heart_beat)
                put = True
            except queue.Full:
                pass


class _Worker(metaclass=abc.ABCMeta):

    def __init__(
        self,
        name: Optional[str],
        i: int,
        work_function: Callable,
        in_q: Union[queue.Queue, multiprocessing.Queue],
        out_q: Union[queue.Queue, multiprocessing.Queue],
        heart_beat: float
    ):
        """
        Individual worker agent.

        :param name: Optional name for this worker.
        :param i: The integer index, >= 0, of this worker among active workers
            for this parallel iteration task.
        :param work_function: Callable function to invoke which generates some
            result value.
        :param in_q: Queue to draw work function input parameters from.
        :param out_q: Queue to output work results, or triggered exceptions,
            to.
        :param heart_beat: How long in seconds we wait when polling for data on
            the `in_q`, as well as for result put attempts into `out_q`, before
            momentarily giving up to allow a cycle of the loop. This is
            important in allowing an external signal to indicate we should stop
            working (prevents hanging on queue interactions).
        """
        self._l_prefix: str = f"[Worker{(name and f'::{name}') or ''}::#{int(i)}]"

        self.i = i
        self.work_function = work_function
        self.in_q = in_q
        self.out_q = out_q
        self.heart_beat = heart_beat
        LOG.log(1, f"{self._l_prefix} Making process worker ({str(in_q)}, {str(out_q)})")

        self._stop_event = self._make_event()

    @classmethod
    @abc.abstractmethod
    def _make_event(cls) -> Union[threading.Event, multiprocessing.synchronize.Event]:
        """
        Generate an event type instance appropriate for the type of worker
        sub-classed.
        """
        raise NotImplementedError()

    def stop(self) -> None:
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        """
        Perform work function on available data in the input queue.
        """
        l_prefix = self._l_prefix
        try:
            packet = self.q_get()
            while not self.stopped():
                if _is_terminal(packet):
                    LOG.log(1, f"{l_prefix} sending terminal")
                    self.q_put(packet)
                    self.stop()
                elif isinstance(packet[0], Exception):
                    # Pass exception along
                    self.q_put(packet)
                    self.stop()
                else:
                    i, args = packet
                    result = self.work_function(*args)
                    self.q_put((i, result))
                    packet = self.q_get()
        # Transport back any exceptions raised
        except (Exception, KeyboardInterrupt) as ex:
            LOG.warning(f"{l_prefix} Caught exception {type(ex)}")
            self.q_put((ex, traceback.format_exc()))
            self.stop()
        except BaseException as ex:
            # Some exotic error occurred (can only be systemExit at this
            # point?). Register stopping and re-raise.
            LOG.log(1, f"Exotic error {type(ex)}: {ex}")
            self.stop()
            raise
        finally:
            LOG.log(1, f"{l_prefix} Closing")

    def q_get(self) -> Any:
        """
        Try to get a value from the queue while keeping an eye out for an exit
        request.

        :return: next value on the input queue
        """
        while not self.stopped():
            try:
                return self.in_q.get(timeout=self.heart_beat)
            except queue.Empty:
                pass

    def q_put(self, val: Any) -> None:
        """
        Try to put the given value into the output queue while keeping an eye
        out for an exit request.

        :param val: value to put into the output queue.

        """
        put = False
        while not put and not self.stopped():
            try:
                self.out_q.put(val, timeout=self.heart_beat)
                put = True
            except queue.Full:
                pass


class _WorkerProcess (_Worker, multiprocessing.Process):

    def __init__(
        self,
        name: Optional[str],
        i: int,
        work_function: Callable,
        in_q: Union[queue.Queue, multiprocessing.Queue],
        out_q: Union[queue.Queue, multiprocessing.Queue],
        heart_beat: float
    ):
        """
        Constructor override to include multiprocessing.Process constructor
        super construction. See `_Worker` constructor doc-string for parameter
        documentation.
        """
        multiprocessing.Process.__init__(self)
        _Worker.__init__(self, name, i, work_function, in_q, out_q, heart_beat)

    @classmethod
    def _make_event(cls) -> multiprocessing.synchronize.Event:
        return multiprocessing.Event()

    # The inheritance order should be sufficient to ensure the `_Worker.run`
    # method is used instead of `multiprocessing.Process.run`, but we are
    # explicit here just to be sure.
    run = _Worker.run


class _WorkerThread (_Worker, threading.Thread):

    def __init__(
        self,
        name: Optional[str],
        i: int,
        work_function: Callable,
        in_q: Union[queue.Queue, multiprocessing.Queue],
        out_q: Union[queue.Queue, multiprocessing.Queue],
        heart_beat: float
    ):
        """
        Constructor override to include threading.Thread constructor super
        construction. See `_Worker` constructor doc-string for parameter
        documentation.
        """
        threading.Thread.__init__(self)
        _Worker.__init__(self, name, i, work_function, in_q, out_q, heart_beat)

    @classmethod
    def _make_event(cls) -> threading.Event:
        return threading.Event()

    # The inheritance order should be sufficient to ensure the `_Worker.run`
    # method is used instead of `threading.Thread.run`, but we are  explicit
    # here just to be sure.
    run = _Worker.run
