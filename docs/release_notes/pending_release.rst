Pending Release Notes
=====================

Updates / New Features
----------------------

* Added official Python 3.14 support and testing, including the PyTorch extra,
  and updated the test tools and dependency lock for compatibility.

* Updated multiprocessing tests and documentation for Python 3.14's default
  forkserver start method.

* Required SMQTK-Core 0.22.0, SMQTK-Dataprovider 0.20.0, and SMQTK-Image-IO
  0.19.0 for Python 3.14 support and NumPy 2 compatibility.

* Declared NumPy compatibility minimums explicitly for Python 3.10–3.12,
  3.13, and 3.14, while allowing NumPy 1 or 2 on Python 3.10–3.12.

Fixes
-----

* Raised the optional PyTorch and Torchvision minimums to patched compatible
  releases, using the same requirements on all supported Python versions.
  Excluded Python 3.14.1 to match Torchvision's compatibility metadata.

* Updated vulnerable locked dependencies to patched releases and raised the
  pytest and IPython minimums to exclude vulnerable releases.

* Corrected typing for the optional PyTorch import.

* Used Amazon ECR Public's mirror of Docker's official Python images in CI to
  avoid Docker Hub's unauthenticated pull rate limit.
