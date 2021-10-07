Pending Release Notes
=====================

Updates / New Features
----------------------

Utilities

* ``parallel_map`` learned to retain to feeder thread and workers until
  iterator completion via "master stop" mechanic. This allows the mechanism to
  now support PyTorch tensors in multiprocessing mode.

Fixes
-----
