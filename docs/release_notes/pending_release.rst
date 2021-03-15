SMQTK Pending Release Notes
===========================


Updates / New Features
----------------------

CI

* Added actions workflow for CI on GitHub.

Misc.

* Updated various type annotations for type-checking compliance.

* Updated to use now publicly available ``smqtk-dataprovider`` package from
  PYPI.


Fixes
-----

CI

* Fix other LGTM warnings.

Descriptor Element

* Removed old ``elements_to_matrix`` utility function, replacing it's use with
  the appropriate invocation of :func:`smqtk.utils.parallel.parallel_map`.

Descriptor Set

* Memory

  * Fixed issue with iter* methods not returning *Iterator* types, which
    specifically caused an issue with `iterdescriptors` as it is used in the
    parent-class definition of ``__iter__``, which requires that an
    iterator-type be returned.

Misc.

* Fixed issue with packages specifier in ``setup.py`` where it was only
  excluding the top-level ``tests`` module but including the rest. Fixed to
  only explicitly include the ``smqtk_descriptors`` package and submodules.

* Fixed issues with type checking mypy tests exposed with more strict settings.
