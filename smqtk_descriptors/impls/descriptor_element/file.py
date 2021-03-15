import os.path as osp
from typing import Any, Dict, Hashable, Mapping, Optional

import numpy

from smqtk_dataprovider.utils.file import safe_create_dir
from smqtk_dataprovider.utils.string import partition_string
from smqtk_descriptors import DescriptorElement


class DescriptorFileElement (DescriptorElement):  # lgtm [py/missing-equals]
    """
    File-based storage of descriptor element.

    When initialized, saves uuid and vector as a serialized pickle-file and
    numpy-format npy file, respectively. These are named according to the string
    representation of the uuid object provided. These are then loaded from disk
    when the ``uuid`` or ``vector`` methods are called. This is in turn slower
    performance wise than ``MemoryElement``, however RAM consumption will be
    lower for large number of elements that would otherwise exceed RAM storage
    space.

    """

    @classmethod
    def is_usable(cls) -> bool:
        return True

    def __init__(
        self,
        type_str: str,
        uuid: Hashable,
        save_dir: str,
        subdir_split: Optional[int] = None
    ):
        """
        Initialize a file-base descriptor element.

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :param uuid: uuid for this descriptor
        :param save_dir: Directory to save this element's contents. If this path
            is relative, we interpret as relative to the current working
            directory.
        :param subdir_split: If a positive integer and greater than 1, this will
            cause us to store the vector file in a subdirectory under the
            ``save_dir`` based on our ``uuid``. The integer value specifies the
            number of splits that we will make in the stringification of this
            descriptor's UUID. The last split component is left off when
            determining the save directory (thus the >1 above).

            Dashes are stripped from this string (as would happen if given an
            uuid.UUID instance as the uuid element).

        """
        super(DescriptorFileElement, self).__init__(type_str, uuid)
        self._save_dir = osp.abspath(osp.expanduser(save_dir))
        self._subdir_split = subdir_split

        # Generate filepath from parameters
        if self._subdir_split and int(self._subdir_split) > 1:
            # TODO: If uuid is an integer, create string with left-padded 0's
            #       to expand out the "length" before partitioning.
            save_dir = osp.join(
                self._save_dir,
                *partition_string(str(self.uuid()).replace('-', ''),
                                  int(self._subdir_split))[:-1]
            )
        else:
            save_dir = self._save_dir
        self._vec_filepath = osp.join(save_dir,
                                      "%s.%s.vector.npy" % (self.type(),
                                                            str(self.uuid())))

    def __getstate__(self) -> Dict[str, Any]:
        state = super(DescriptorFileElement, self).__getstate__()
        state.update({
            '_save_dir': self._save_dir,
            '_subdir_split': self._subdir_split,
            '_vec_filepath': self._vec_filepath,
        })
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        super(DescriptorFileElement, self).__setstate__(state)
        self._save_dir = state['_save_dir']
        self._subdir_split = state['_subdir_split']
        self._vec_filepath = state['_vec_filepath']

    def get_config(self) -> Dict[str, Any]:
        return {
            "save_dir": self._save_dir,
            'subdir_split': self._subdir_split
        }

    def has_vector(self) -> bool:
        """
        :return: Whether or not this container current has a descriptor vector
            stored.
        :rtype: bool
        """
        return osp.isfile(self._vec_filepath)

    def vector(self) -> Optional[numpy.ndarray]:
        """
        :return: Get the stored descriptor vector as a numpy array. This returns
            None of there is no vector stored in this container.
        :rtype: numpy.core.multiarray.ndarray or None
        """
        # TODO: Load as memmap?
        #       i.e. modifications by user to vector will be reflected on disk.
        if self.has_vector():
            return numpy.load(self._vec_filepath)
        else:
            return None

    def set_vector(self, new_vec: numpy.ndarray) -> "DescriptorElement":
        """
        Set the contained vector.

        If this container already stores a descriptor vector, this will
        overwrite it.

        :param new_vec: New vector to contain.
        :type new_vec: numpy.core.multiarray.ndarray

        :returns: Self.
        :rtype: DescriptorFileElement

        """
        safe_create_dir(osp.dirname(self._vec_filepath))
        numpy.save(self._vec_filepath, new_vec)
        return self
