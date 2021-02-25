from io import BytesIO

import numpy

from smqtk_descriptors import DescriptorElement


class DescriptorMemoryElement (DescriptorElement):
    """
    In-memory representation of descriptor elements. Stored vectors are
    effectively immutable.

    Example
    -------
    >>> self = DescriptorMemoryElement('random', 0)
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, type_str, uuid):
        super(DescriptorMemoryElement, self).__init__(type_str, uuid)
        self.__v = None

    @classmethod
    def _get_many_vectors(cls, descriptors):
        # Memory elements are super simple.
        for d in descriptors:
            yield d.uuid(), d.vector()

    def __getstate__(self):
        state = super(DescriptorMemoryElement, self).__getstate__()
        # save vector as binary string
        b = BytesIO()
        # noinspection PyTypeChecker
        numpy.save(b, self.vector())
        state['v'] = b.getvalue()
        return state

    def __setstate__(self, state):
        # Handle the previous state format:
        if isinstance(state, tuple):
            self._type_label = state[0]
            self._uuid = state[1]
            b = BytesIO(state[2])
        else:  # dictionary
            super(DescriptorMemoryElement, self).__setstate__(state)
            b = BytesIO(state['v'])
        self.__v = numpy.load(b)

    def get_config(self):
        """
        :return: JSON type compliant configuration dictionary.
        :rtype: dict
        """
        return {}

    def has_vector(self):
        """
        :return: Whether or not this container current has a descriptor vector
            stored.
        :rtype: bool
        """
        return self.__v is not None

    def vector(self):
        """
        Implementation Note
        -------------------
        A copy of the internally stored vector is returned.

        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.core.multiarray.ndarray or None

        """
        # Copy if storing an array, otherwise return the None value
        if self.__v is not None:
            return numpy.copy(self.__v)
        return None

    def set_vector(self, new_vec):
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        ``new_vec`` may be None, which clears this descriptor's vector from the
        cache.

        Implementation Note
        -------------------
        This implementation copies input arrays before storage to mimic
        immutability.

        :param new_vec: New vector to contain.
        :type new_vec: numpy.core.multiarray.ndarray | tuple | list | None

        :returns: Self.
        :rtype: DescriptorMemoryElement

        """
        # Copy a non-None value given, otherwise stay None
        if new_vec is not None:
            self.__v = numpy.copy(new_vec)
        else:
            self.__v = None
        return self
