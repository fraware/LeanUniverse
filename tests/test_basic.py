"""Basic tests for LeanUniverse."""

import pytest
from datetime import datetime

from lean_universe.models.repository import RepositoryInfo, RepositoryStatus
from lean_universe.config import get_config


class TestRepositoryInfo:
    """Test the RepositoryInfo model."""

    def test_repository_info_creation(self):
        """Test creating a RepositoryInfo instance."""
        repo_info = RepositoryInfo(
            url="https://github.com/test/repo",
            name="repo",
            full_name="test/repo",
            description="Test repository",
            language="Lean",
            stars=100,
            forks=10,
            created_at=datetime(2023, 1, 1),
            updated_at=datetime(2024, 1, 1),
            license="MIT",
            default_branch="main",
            is_fork=False,
            size=1000,
        )

        assert repo_info.url == "https://github.com/test/repo"
        assert repo_info.name == "repo"
        assert repo_info.full_name == "test/repo"
        assert repo_info.status == RepositoryStatus.DISCOVERED

    def test_repository_info_defaults(self):
        """Test RepositoryInfo with default values."""
        repo_info = RepositoryInfo(
            url="https://github.com/test/repo",
            name="repo",
            full_name="test/repo",
            created_at=datetime(2023, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )

        assert repo_info.description == ""
        assert repo_info.language == "Unknown"
        assert repo_info.stars == 0
        assert repo_info.forks == 0
        assert repo_info.license is None
        assert repo_info.default_branch == "main"
        assert repo_info.is_fork is False
        assert repo_info.size == 0

    def test_repository_info_serialization(self):
        """Test RepositoryInfo serialization."""
        repo_info = RepositoryInfo(
            url="https://github.com/test/repo",
            name="repo",
            full_name="test/repo",
            created_at=datetime(2023, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )

        data = repo_info.model_dump()
        assert "url" in data
        assert "name" in data
        assert "full_name" in data
        assert "status" in data
        assert data["status"] == "discovered"


class TestRepositoryStatus:
    """Test the RepositoryStatus enum."""

    def test_repository_status_values(self):
        """Test RepositoryStatus enum values."""
        assert RepositoryStatus.DISCOVERED == "discovered"
        assert RepositoryStatus.CLONED == "cloned"
        assert RepositoryStatus.PROCESSING == "processing"
        assert RepositoryStatus.PROCESSED == "processed"
        assert RepositoryStatus.FAILED == "failed"
        assert RepositoryStatus.SKIPPED == "skipped"

    def test_repository_status_comparison(self):
        """Test RepositoryStatus comparison."""
        status1 = RepositoryStatus.DISCOVERED
        status2 = RepositoryStatus.CLONED

        assert status1 != status2
        assert status1 == RepositoryStatus.DISCOVERED


class TestConfig:
    """Test the configuration system."""

    def test_config_loading(self):
        """Test that configuration can be loaded."""
        config = get_config()
        assert config is not None
        assert hasattr(config, "github")
        assert hasattr(config, "lean_dojo")
        assert hasattr(config, "ml")
        assert hasattr(config, "monitoring")

    def test_config_defaults(self):
        """Test configuration default values."""
        config = get_config()

        # Test GitHub defaults
        assert config.github.max_concurrent_requests > 0
        assert config.github.rate_limit_retry_attempts > 0

        # Test LeanDojo defaults
        assert config.lean_dojo.max_num_procs > 0
        assert config.lean_dojo.timeout > 0

        # Test ML defaults
        assert config.ml.batch_size > 0
        assert config.ml.precision in ["float16", "float32", "float64"]

        # Test monitoring defaults
        assert config.monitoring.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]


class TestVersion:
    """Test version information."""

    def test_version_import(self):
        """Test that version can be imported."""
        from lean_universe import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        from lean_universe import __version__

        # Should be in format like "0.2.0"
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(part.isdigit() for part in parts[:2])


def test_imports():
    """Test that all main modules can be imported."""
    # Test core imports
    from lean_universe import get_config, AsyncRepositoryManager, RepositoryInfo, RepositoryStatus

    assert get_config is not None
    assert AsyncRepositoryManager is not None
    assert RepositoryInfo is not None
    assert RepositoryStatus is not None
