SMQTK Pending Release Notes
===========================


Updates / New Features
----------------------

Misc.

* Updated various type annotations for type-checking compliance.


Fixes
-----

Descriptor Set

* Memory

  * Fixed issue with iter* methods not returning *Iterator* types, which
    specifically caused an issue with `iterdescriptors` as it is used in the
    parent-class definition of ``__iter__``, which requires that an
    iterator-type be returned.
