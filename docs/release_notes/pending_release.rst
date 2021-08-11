Pending Release Notes
=====================


Updates / New Features
----------------------

Deprecations

* Deprecated and renamed various functions with `iter*` prefix because this is
  no longer a standard. Deprecation warnings were added to preserve usability.

* Removed `type_str` from `descriptor_element` interface and impls because the
  type-string attribute isn't utilized.

Utils

* More usefully type and annotate
  :func:`smqtk_descriptors.utils.parallel.parallel_map` to pass through the
  input callable's annotated return type as the iteration output of the
  returned :class:`~smqtk_descriptors.utils.parallel.ParallelResultsIterator`
  instance.

Features

* Created :class:`~smqtk_descriptors.interfaces.image_descriptor_generator.ImageDescriptorGenerator`
  And added an implementation :class:`smqtk_descriptors.impls.descriptor_generator.image_descriptor_generator_wrapper.ImageDescriptorGeneratorWrapper`
  for the purposes of handling image matrices directly.

Fixes
-----
