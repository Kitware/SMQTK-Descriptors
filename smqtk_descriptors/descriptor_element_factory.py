from typing import Any, Dict, Hashable, Type, TypeVar

from smqtk_core import Configurable
from smqtk_core.configuration import (
    cls_conf_from_config_dict,
    cls_conf_to_config_dict,
    make_default_config,
)
from smqtk_core.dict import merge_dict

from smqtk_descriptors.interfaces.descriptor_element import DescriptorElement


T = TypeVar("T", bound="DescriptorElementFactory")


class DescriptorElementFactory (Configurable):
    """
    Factory class for producing DescriptorElement instances of a specified type
    and configuration.
    """

    def __init__(self, d_type: Type[DescriptorElement], type_config: Dict[str, Any]):
        """
        Initialize the factory to produce DescriptorElement instances of the
        given type and configuration.

        :param d_type: Type of descriptor element this factory should produce.
        :param type_config: Initialization parameter dictionary that should
            contain all additional construction parameters for the provided type
            except for the expected `type_str` and `uuid` arguments that should
            be the first and second positional arguments respectively.
        """
        self._d_type = d_type
        self._d_type_config = type_config

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Generate and return a default configuration dictionary for this class.
        This will be primarily used for generating what the configuration
        dictionary would look like for this class without instantiating it.

        It is not be guaranteed that the configuration dictionary returned
        from this method is valid for construction of an instance of this class.

        :return: Default configuration dictionary for the class.
        """
        return make_default_config(DescriptorElement.get_impls())

    @classmethod
    def from_config(
        cls: Type[T],
        config_dict: Dict,
        merge_default: bool = True
    ) -> T:
        """
        Instantiate a new instance of this class given the configuration
        JSON-compliant dictionary encapsulating initialization arguments.

        This method should not be called via super unless and instance of the
        class is desired.

        :param config_dict: JSON compliant dictionary encapsulating
            a configuration.
        :param merge_default: Merge the given configuration on top of the
            default provided by ``get_default_config``.

        :return: Constructed instance from the provided config.
        """
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        de_type, de_conf = cls_conf_from_config_dict(
            config_dict, DescriptorElement.get_impls()
        )
        return cls(de_type, de_conf)

    def get_config(self) -> Dict[str, Any]:
        return cls_conf_to_config_dict(self._d_type, self._d_type_config)

    def new_descriptor(self, type_str: str, uuid: Hashable) -> DescriptorElement:
        """
        Create a new DescriptorElement instance of the configured implementation

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :param uuid: UUID to associate with the descriptor

        :return: New DescriptorElement instance
        """
        return self._d_type.from_config(self._d_type_config, type_str, uuid)

    def __call__(self, type_str: str, uuid: Hashable) -> DescriptorElement:
        """
        Create a new DescriptorElement instance of the configured implementation

        :param type_str: Type of descriptor. This is usually the name of the
            content descriptor that generated this vector.
        :param uuid: UUID to associate with the descriptor

        :return: New DescriptorElement instance
        """
        return self.new_descriptor(type_str, uuid)
