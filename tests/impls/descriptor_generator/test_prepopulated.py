import pytest
from smqtk_descriptors.impls.descriptor_generator.prepopulated import (
    PrePopulatedDescriptorGenerator,
)


def test_valid_content_types() -> None:
    """
    Tests that valid_content_types() returns an empty set.
    """
    generator = PrePopulatedDescriptorGenerator()
    assert generator.valid_content_types() == set()


def test_generate_arrays() -> None:
    """
    Tests that _generate_arrays() method raises AssertionError.
    """
    generator = PrePopulatedDescriptorGenerator()

    with pytest.raises(AssertionError):
        generator._generate_arrays([])


def test_get_config() -> None:
    """
    Tests that get_config() returns an empty dictionary.
    """
    generator = PrePopulatedDescriptorGenerator()
    assert generator.get_config() == {}
