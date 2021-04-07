# SMQTK - Descriptors

## Intent
This package aims to provide interfaces for algorithms and data structures
around computing descriptor vectors from input data.

This package also includes a utility function that can map an arbitrary
function to some set of input iterables, not unlike the python `map`, except
that this version is parallelized across threads or processes. This function
also does not block and may be used for parallelized stream processing.

## Documentation
You can build the sphinx documentation locally for the most up-to-date
reference:
```bash
# Install dependencies
poetry install
# Navigate to the documentation root.
cd docs
# Build the docs.
poetry run make html
# Open in your favorite browser!
firefox _build/html/index.html
```
