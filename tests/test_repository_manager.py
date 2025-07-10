# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the AsyncRepositoryManager."""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lean_universe.models.repository import RepositoryInfo, RepositoryStatus
from lean_universe.repository.manager import AsyncRepositoryManager


@pytest.fixture
def mock_github_repo():
    """Create a mock GitHub repository."""
    repo = MagicMock()
    repo.html_url = "https://github.com/test/lean-repo"
    repo.name = "lean-repo"
    repo.full_name = "test/lean-repo"
    repo.description = "A test Lean repository"
    repo.language = "Lean"
    repo.stargazers_count = 100
    repo.forks_count = 10
    repo.created_at = datetime(2023, 1, 1)
    repo.updated_at = datetime(2024, 1, 1)
    repo.license = MagicMock()
    repo.license.spdx_id = "MIT"
    repo.default_branch = "main"
    repo.fork = False
    repo.size = 1000
    repo.get_contents.return_value = [
        MagicMock(name="lakefile.lean"),
        MagicMock(name="lean-toolchain"),
    ]
    return repo


@pytest.fixture
def sample_repo_info():
    """Create a sample repository info."""
    return RepositoryInfo(
        url="https://github.com/test/lean-repo",
        name="lean-repo",
        full_name="test/lean-repo",
        description="A test Lean repository",
        language="Lean",
        stars=100,
        forks=10,
        created_at=datetime(2023, 1, 1),
        updated_at=datetime(2024, 1, 1),
        license="MIT",
        default_branch="main",
        is_fork=False,
        size=1000,
        status=RepositoryStatus.DISCOVERED,
    )


class TestAsyncRepositoryManager:
    """Test suite for AsyncRepositoryManager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test repository manager initialization."""
        manager = AsyncRepositoryManager()
        assert manager.config is not None
        assert manager.github is None
        assert manager.session is None
        assert manager.throttler is not None

    @pytest.mark.asyncio
    async def test_manager_context_manager(self):
        """Test repository manager as context manager."""
        async with AsyncRepositoryManager() as manager:
            assert manager.config is not None
            # Should be initialized after entering context
            assert manager.github is not None
            assert manager.session is not None

    @pytest.mark.asyncio
    async def test_discover_repositories_empty(self):
        """Test repository discovery with no results."""
        async with AsyncRepositoryManager() as manager:
            repos = await manager.discover_repositories(max_repos=0, include_repos=[])
            assert len(repos) == 0

    @pytest.mark.asyncio
    async def test_discover_repositories_with_included(self, mock_github_repo):
        """Test repository discovery with included repositories."""
        with patch.object(AsyncRepositoryManager, "_get_repository_info") as mock_get_info:
            mock_get_info.return_value = RepositoryInfo(
                url=mock_github_repo.html_url,
                name=mock_github_repo.name,
                full_name=mock_github_repo.full_name,
                description=mock_github_repo.description,
                language=mock_github_repo.language,
                stars=mock_github_repo.stargazers_count,
                forks=mock_github_repo.forks_count,
                created_at=mock_github_repo.created_at,
                updated_at=mock_github_repo.updated_at,
                license=mock_github_repo.license.spdx_id,
                default_branch=mock_github_repo.default_branch,
                is_fork=mock_github_repo.fork,
                size=mock_github_repo.size,
                status=RepositoryStatus.DISCOVERED,
            )

            async with AsyncRepositoryManager() as manager:
                repos = await manager.discover_repositories(include_repos=[mock_github_repo.html_url])
                assert len(repos) == 1
                assert repos[0].full_name == mock_github_repo.full_name

    @pytest.mark.asyncio
    async def test_validate_repository_valid(self, mock_github_repo):
        """Test repository validation with valid repository."""
        async with AsyncRepositoryManager() as manager:
            is_valid = await manager._validate_repository(mock_github_repo)
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_repository_fork(self, mock_github_repo):
        """Test repository validation with fork repository."""
        mock_github_repo.fork = True

        async with AsyncRepositoryManager() as manager:
            is_valid = await manager._validate_repository(mock_github_repo)
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_repository_missing_lakefile(self, mock_github_repo):
        """Test repository validation with missing lakefile."""
        mock_github_repo.get_contents.return_value = [
            MagicMock(name="lean-toolchain"),
        ]

        async with AsyncRepositoryManager() as manager:
            is_valid = await manager._validate_repository(mock_github_repo)
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_repository_missing_toolchain(self, mock_github_repo):
        """Test repository validation with missing toolchain."""
        mock_github_repo.get_contents.return_value = [
            MagicMock(name="lakefile.lean"),
        ]

        async with AsyncRepositoryManager() as manager:
            is_valid = await manager._validate_repository(mock_github_repo)
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_repository_lean3(self, mock_github_repo):
        """Test repository validation with Lean3 repository."""
        mock_github_repo.get_contents.return_value = [
            MagicMock(name="lakefile.lean"),
            MagicMock(name="lean-toolchain"),
            MagicMock(name="leanpkg.toml"),
        ]

        async with AsyncRepositoryManager() as manager:
            is_valid = await manager._validate_repository(mock_github_repo)
            assert is_valid is False

    @pytest.mark.asyncio
    async def test_convert_github_repo_to_info(self, mock_github_repo):
        """Test converting GitHub repository to RepositoryInfo."""
        async with AsyncRepositoryManager() as manager:
            repo_info = await manager._convert_github_repo_to_info(mock_github_repo)

            assert repo_info is not None
            assert repo_info.url == mock_github_repo.html_url
            assert repo_info.name == mock_github_repo.name
            assert repo_info.full_name == mock_github_repo.full_name
            assert repo_info.description == mock_github_repo.description
            assert repo_info.language == mock_github_repo.language
            assert repo_info.stars == mock_github_repo.stargazers_count
            assert repo_info.forks == mock_github_repo.forks_count
            assert repo_info.license == mock_github_repo.license.spdx_id
            assert repo_info.default_branch == mock_github_repo.default_branch
            assert repo_info.is_fork == mock_github_repo.fork
            assert repo_info.size == mock_github_repo.size
            assert repo_info.status == RepositoryStatus.DISCOVERED

    @pytest.mark.asyncio
    async def test_convert_github_repo_to_info_invalid(self, mock_github_repo):
        """Test converting invalid GitHub repository."""
        mock_github_repo.fork = True  # Make it invalid

        async with AsyncRepositoryManager() as manager:
            repo_info = await manager._convert_github_repo_to_info(mock_github_repo)
            assert repo_info is None

    @pytest.mark.asyncio
    async def test_clone_repositories_empty(self):
        """Test cloning with empty repository list."""
        async with AsyncRepositoryManager() as manager:
            repos = await manager.clone_repositories([])
            assert len(repos) == 0

    @pytest.mark.asyncio
    async def test_clone_repositories_success(self, sample_repo_info, tmp_path):
        """Test successful repository cloning."""
        with patch("lean_universe.repository.manager.porcelain.clone") as mock_clone:
            mock_clone.return_value = None

            async with AsyncRepositoryManager() as manager:
                manager.config.repos_dir = tmp_path

                repos = await manager.clone_repositories([sample_repo_info])
                assert len(repos) == 1
                assert repos[0].status == RepositoryStatus.CLONED
                assert repos[0].local_path is not None

    @pytest.mark.asyncio
    async def test_clone_repositories_failure(self, sample_repo_info, tmp_path):
        """Test repository cloning failure."""
        with patch("lean_universe.repository.manager.porcelain.clone") as mock_clone:
            mock_clone.side_effect = Exception("Clone failed")

            async with AsyncRepositoryManager() as manager:
                manager.config.repos_dir = tmp_path

                repos = await manager.clone_repositories([sample_repo_info])
                assert len(repos) == 1
                assert repos[0].status == RepositoryStatus.FAILED
                assert repos[0].error_message == "Clone failed"

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting manager statistics."""
        async with AsyncRepositoryManager() as manager:
            stats = manager.get_statistics()

            assert "processed_count" in stats
            assert "failed_count" in stats
            assert "failed_repos" in stats
            assert "rate_limit_reset" in stats

            assert isinstance(stats["processed_count"], int)
            assert isinstance(stats["failed_count"], int)
            assert isinstance(stats["failed_repos"], dict)

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test rate limit handling."""
        async with AsyncRepositoryManager() as manager:
            # Mock rate limit exceeded
            with patch.object(manager.github, "get_rate_limit") as mock_rate_limit:
                mock_rate_limit.return_value.core.reset = datetime(2024, 1, 1)

                with patch("asyncio.sleep") as mock_sleep:
                    await manager._handle_rate_limit()
                    mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_needs_update_true(self, sample_repo_info, tmp_path):
        """Test repository update detection when update is needed."""
        with patch("lean_universe.repository.manager.Repo") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.head.return_value = b"old_commit"
            mock_repo_class.return_value = mock_repo

            sample_repo_info.latest_commit = "new_commit"

            async with AsyncRepositoryManager() as manager:
                needs_update = await manager._needs_update(tmp_path, sample_repo_info)
                assert needs_update is True

    @pytest.mark.asyncio
    async def test_needs_update_false(self, sample_repo_info, tmp_path):
        """Test repository update detection when no update is needed."""
        with patch("lean_universe.repository.manager.Repo") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo.head.return_value = b"same_commit"
            mock_repo_class.return_value = mock_repo

            sample_repo_info.latest_commit = "same_commit"

            async with AsyncRepositoryManager() as manager:
                needs_update = await manager._needs_update(tmp_path, sample_repo_info)
                assert needs_update is False

    @pytest.mark.asyncio
    async def test_update_repository_success(self, sample_repo_info, tmp_path):
        """Test successful repository update."""
        with patch("lean_universe.repository.manager.Repo") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            with patch("lean_universe.repository.manager.porcelain.pull") as mock_pull:
                mock_pull.return_value = None

                async with AsyncRepositoryManager() as manager:
                    await manager._update_repository(tmp_path, sample_repo_info)
                    mock_pull.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_repository_failure(self, sample_repo_info, tmp_path):
        """Test repository update failure."""
        with patch("lean_universe.repository.manager.Repo") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            with patch("lean_universe.repository.manager.porcelain.pull") as mock_pull:
                mock_pull.side_effect = Exception("Update failed")

                async with AsyncRepositoryManager() as manager:
                    with pytest.raises(Exception, match="Update failed"):
                        await manager._update_repository(tmp_path, sample_repo_info)


@pytest.mark.asyncio
async def test_repository_manager_context_manager():
    """Test AsyncRepositoryManager as context manager."""
    async with AsyncRepositoryManager() as manager:
        assert manager.github is not None
        assert manager.session is not None

    # After context exit, session should be closed
    assert manager.session.closed


@pytest.mark.asyncio
async def test_repository_manager_error_handling():
    """Test error handling in repository manager."""
    with patch("lean_universe.repository.manager.Github") as mock_github:
        mock_github.side_effect = Exception("GitHub connection failed")

        with pytest.raises(Exception, match="GitHub connection failed"):
            async with AsyncRepositoryManager():
                pass


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
