# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Custom exceptions for LeanUniverse."""


class LeanUniverseError(Exception):
    """Base exception for LeanUniverse."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class RepositoryError(LeanUniverseError):
    """Base exception for repository-related errors."""

    pass


class RepositoryCloneError(RepositoryError):
    """Exception raised when repository cloning fails."""

    pass


class RepositoryValidationError(RepositoryError):
    """Exception raised when repository validation fails."""

    pass


class RateLimitError(LeanUniverseError):
    """Exception raised when API rate limits are exceeded."""

    pass


class ConfigurationError(LeanUniverseError):
    """Exception raised when configuration is invalid."""

    pass


class DatasetError(LeanUniverseError):
    """Base exception for dataset-related errors."""

    pass


class DatasetGenerationError(DatasetError):
    """Exception raised when dataset generation fails."""

    pass


class LeanDojoError(LeanUniverseError):
    """Exception raised when LeanDojo operations fail."""

    pass


class BuildError(LeanUniverseError):
    """Exception raised when Lean4 build operations fail."""

    pass


class ValidationError(LeanUniverseError):
    """Exception raised when data validation fails."""

    pass


class CacheError(LeanUniverseError):
    """Exception raised when cache operations fail."""

    pass


class SecurityError(LeanUniverseError):
    """Exception raised when security checks fail."""

    pass
