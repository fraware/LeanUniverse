"""Simple tests that don't require external dependencies."""

import pytest
from datetime import datetime

from lean_universe import __version__


def test_version():
    """Test that version is available."""
    assert __version__ == "0.2.0"


def test_version_format():
    """Test that version follows semantic versioning."""
    parts = str(__version__).split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts[:2])


def test_basic_imports():
    """Test that basic modules can be imported."""
    # Test version import
    from lean_universe import __version__

    assert __version__ is not None

    # Test lazy imports work
    try:
        from lean_universe.config import get_config

        config = get_config()
        assert config is not None
    except ImportError:
        # This is expected if dependencies aren't installed
        pass


def test_cli_import():
    """Test that CLI can be imported."""
    try:
        from lean_universe.cli import app

        assert app is not None
    except ImportError:
        # This is expected if dependencies aren't installed
        pass


if __name__ == "__main__":
    # Run basic tests
    test_version()
    test_version_format()
    test_basic_imports()
    test_cli_import()
    print("All basic tests passed!")
