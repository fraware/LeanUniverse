# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
LeanUniverse: Modern Lean4 Dataset Management with AI Integration

A revolutionary library for creating comprehensive, AI-ready datasets from Lean4 repositories
with cutting-edge features and modern architecture.
"""

from .__version__ import __version__

__all__ = [
    "__version__",
]


# Lazy imports to avoid dependency issues during basic testing
def get_config():
    """Get the global configuration instance."""
    from .config import get_config as _get_config

    return _get_config()


def AsyncRepositoryManager():
    """Get the AsyncRepositoryManager class."""
    from .repository.manager import AsyncRepositoryManager as _AsyncRepositoryManager

    return _AsyncRepositoryManager


def RepositoryInfo():
    """Get the RepositoryInfo class."""
    from .models.repository import RepositoryInfo as _RepositoryInfo

    return _RepositoryInfo


def RepositoryStatus():
    """Get the RepositoryStatus enum."""
    from .models.repository import RepositoryStatus as _RepositoryStatus

    return _RepositoryStatus
