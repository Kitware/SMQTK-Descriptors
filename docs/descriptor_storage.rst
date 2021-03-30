Descriptor Storage
==================

DescriptorElement Interface
---------------------------
The :class:`.DescriptorElement` interface defines a standard for storing and
retrieving a descriptor vector and it's associated UID.
Descriptors, also known as feature vectors, are defined here as
:class:`numpy.ndarray` instances.
We do not constrain the vector data type at this level.

Descriptor elements are also associated with a UID.
There is no standard for UID generation imposed here and is left to the user
or generating algorithm to define UID attribution.
Generally a UID, or `unique identifier`_, "is an identifier that is guaranteed
to be unique among all identifiers  used for those objects and for a specific
purpose."

These are generally constrained to fit the python `Hashable`_ type definition.

.. _unique identifier: https://en.wikipedia.org/wiki/Unique_identifier
.. _Hashable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable

Storing Many Elements
---------------------
We provide an interface for storing groups of descriptor elements called the
:class:`.DescriptorSet`.
This provides an interface for storing and retrieving sets of
:class:`.DescriptorElement` instances, accessing by UID, and iterating over
contained elements.

Reference
---------

.. autoclass:: smqtk_descriptors.interfaces.descriptor_element.DescriptorElement
   :members:

.. autoclass:: smqtk_descriptors.interfaces.descriptor_set.DescriptorSet
   :members:

.. autoclass:: smqtk_descriptors.descriptor_element_factory.DescriptorElementFactory
   :members:
