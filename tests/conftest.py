"""Pytest configuration and fixtures for LeanUniverse tests."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from lean_universe.config import get_config
from lean_universe.models.repository import RepositoryInfo, RepositoryStatus
from lean_universe.repository.manager import AsyncRepositoryManager


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock configuration for testing."""
    config = MagicMock()
    config.github.access_token = "test_token"
    config.github.max_concurrent_requests = 5
    config.github.rate_limit_delay = 0.1
    config.lean_dojo.max_num_procs = 4
    config.lean_dojo.timeout = 30
    config.ml.device = "cpu"
    config.ml.batch_size = 16
    config.monitoring.enable_prometheus = False
    config.monitoring.log_level = "DEBUG"
    return config


@pytest.fixture
def sample_repository_info() -> RepositoryInfo:
    """Create a sample repository info for testing."""
    return RepositoryInfo(
        url="https://github.com/test/lean-repo",
        name="lean-repo",
        full_name="test/lean-repo",
        description="A test Lean4 repository",
        language="Lean",
        stars=100,
        forks=10,
        created_at=datetime(2023, 1, 1),
        updated_at=datetime(2024, 1, 1),
        license="MIT",
        default_branch="main",
        is_fork=False,
        size=1024,
        status=RepositoryStatus.DISCOVERED,
    )


@pytest.fixture
def mock_github_client() -> AsyncMock:
    """Create a mock GitHub client for testing."""
    client = AsyncMock()

    # Mock repository search
    client.search_repositories.return_value = {
        "total_count": 1,
        "items": [
            {
                "html_url": "https://github.com/test/lean-repo",
                "name": "lean-repo",
                "full_name": "test/lean-repo",
                "description": "A test Lean4 repository",
                "language": "Lean",
                "stargazers_count": 100,
                "forks_count": 10,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "license": {"spdx_id": "MIT"},
                "default_branch": "main",
                "fork": False,
                "size": 1024,
            }
        ],
    }

    return client


@pytest.fixture
async def async_repository_manager() -> AsyncGenerator[AsyncRepositoryManager, None]:
    """Create an async repository manager for testing."""
    async with AsyncRepositoryManager() as manager:
        yield manager


@pytest.fixture
def mock_lean_dojo() -> AsyncMock:
    """Create a mock LeanDojo client for testing."""
    client = AsyncMock()
    client.build_repo.return_value = {
        "success": True,
        "build_time": 10.5,
        "theorems": 50,
        "proofs": 45,
    }
    return client


@pytest.fixture
def mock_redis_client() -> AsyncMock:
    """Create a mock Redis client for testing."""
    client = AsyncMock()
    client.get.return_value = None
    client.set.return_value = True
    client.exists.return_value = False
    return client


@pytest.fixture
def mock_database_session() -> AsyncMock:
    """Create a mock database session for testing."""
    session = AsyncMock()
    session.add.return_value = None
    session.commit.return_value = None
    session.rollback.return_value = None
    session.close.return_value = None
    return session


@pytest.fixture
def mock_prometheus_metrics() -> MagicMock:
    """Create mock Prometheus metrics for testing."""
    metrics = MagicMock()
    metrics.counter.return_value = MagicMock()
    metrics.gauge.return_value = MagicMock()
    metrics.histogram.return_value = MagicMock()
    return metrics


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger for testing."""
    logger = MagicMock()
    logger.info.return_value = None
    logger.warning.return_value = None
    logger.error.return_value = None
    logger.debug.return_value = None
    return logger


# Test data fixtures
@pytest.fixture
def sample_lean_files() -> list[dict]:
    """Sample Lean file data for testing."""
    return [
        {
            "path": "src/example.lean",
            "content": """
import Mathlib.Data.Nat.Basic

theorem example_theorem (n : ℕ) : n + 0 = n := by
  rw [Nat.add_zero]
""",
            "theorems": 1,
            "proofs": 1,
        },
        {
            "path": "src/another.lean",
            "content": """
import Mathlib.Algebra.Ring.Basic

lemma another_lemma (a b : ℕ) : a + b = b + a := by
  exact Nat.add_comm a b
""",
            "theorems": 1,
            "proofs": 1,
        },
    ]


@pytest.fixture
def sample_dataset() -> dict:
    """Sample dataset for testing."""
    return {
        "metadata": {
            "version": "0.2.0",
            "created_at": "2024-01-01T00:00:00Z",
            "repository_count": 1,
            "total_theorems": 2,
            "total_proofs": 2,
        },
        "repositories": [
            {
                "url": "https://github.com/test/lean-repo",
                "name": "lean-repo",
                "theorems": [
                    {
                        "name": "example_theorem",
                        "statement": "∀ (n : ℕ), n + 0 = n",
                        "proof": "rw [Nat.add_zero]",
                        "file": "src/example.lean",
                        "line": 4,
                    }
                ],
            }
        ],
    }


# CLI testing fixtures
@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    from typer.testing import CliRunner

    runner = CliRunner()
    return runner


@pytest.fixture
def mock_environment(monkeypatch: pytest.MonkeyPatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("GITHUB__ACCESS_TOKEN", "test_token")
    monkeypatch.setenv("GITHUB__MAX_CONCURRENT_REQUESTS", "5")
    monkeypatch.setenv("MONITORING__LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MONITORING__ENABLE_PROMETHEUS", "false")
