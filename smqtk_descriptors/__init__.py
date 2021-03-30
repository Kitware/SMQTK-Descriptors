import pkg_resources

from .descriptor_element_factory import DescriptorElementFactory  # noqa: F401
from .interfaces.descriptor_element import DescriptorElement  # noqa: F401
from .interfaces.descriptor_set import DescriptorSet  # noqa: F401
from .interfaces.descriptor_generator import DescriptorGenerator  # noqa: F401


# It is known that this will fail if this package is not "installed" in the
# current environment. Additional support is pending defined use-case-driven
# requirements.
__version__ = pkg_resources.get_distribution(__name__).version
