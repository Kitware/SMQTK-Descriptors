Parallelism
===========
This package provides a parallel work mapping function called
:meth:`~smqtk_descriptors.utils.parallel.parallel_map` to allow for stream
parallelism of arbitrary work functions using multi-threading or
multi-processing.
Which this method of parallelism is not universally the most efficient, it is
intended to operate at it's best under situations where work functions are
dynamic or parallelism with streaming input is required.

Reference
---------
.. autofunction:: smqtk_descriptors.utils.parallel.parallel_map

.. autoclass:: smqtk_descriptors.utils.parallel.ParallelResultsIterator
   :members:
