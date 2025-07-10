# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Modern async repository management for LeanUniverse."""

import asyncio
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiofiles
import aiohttp
import structlog
from asyncio_throttle import Throttler
from dulwich import porcelain
from dulwich.repo import Repo
from github import Auth, Github, RateLimitExceededException, Repository
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lean_universe.config import get_config
from lean_universe.models.repository import RepositoryInfo, RepositoryStatus
from lean_universe.utils.exceptions import (
    RepositoryCloneError,
    RepositoryValidationError,
    RateLimitError,
)

logger = structlog.get_logger(__name__)


class AsyncRepositoryManager:
    """Modern async repository manager with advanced features."""

    def __init__(self) -> None:
        self.config = get_config()
        self.github: Optional[Github] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.throttler = Throttler(rate_limit=self.config.github.max_concurrent_requests)
        self._processed_repos: Set[str] = set()
        self._failed_repos: Dict[str, str] = {}
        self._rate_limit_reset: Optional[datetime] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()

    async def _setup(self) -> None:
        """Setup async resources."""
        # Initialize GitHub client
        if self.config.github.access_token:
            auth = Auth.Token(self.config.github.access_token)
            self.github = Github(auth=auth)
        else:
            self.github = Github()

        # Initialize HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)

        # Create necessary directories
        self.config.repos_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    async def _cleanup(self) -> None:
        """Cleanup async resources."""
        if self.session:
            await self.session.close()

    async def discover_repositories(
        self,
        query: str = "lean",
        language: str = "lean",
        max_repos: Optional[int] = None,
        include_repos: Optional[List[str]] = None,
        exclude_repos: Optional[List[str]] = None,
    ) -> List[RepositoryInfo]:
        """Discover Lean repositories asynchronously."""
        if max_repos is None:
            max_repos = self.config.max_num_repos

        include_repos = include_repos or self.config.repos_included
        exclude_repos = exclude_repos or self.config.repos_excluded

        logger.info(
            "Starting repository discovery",
            query=query,
            language=language,
            max_repos=max_repos,
            include_count=len(include_repos),
            exclude_count=len(exclude_repos),
        )

        discovered_repos: List[RepositoryInfo] = []

        # Process explicitly included repositories
        if include_repos:
            async for repo_info in self._process_included_repos(include_repos):
                if len(discovered_repos) >= max_repos:
                    break
                discovered_repos.append(repo_info)

        # Search for additional repositories
        if len(discovered_repos) < max_repos and self.github:
            remaining_count = max_repos - len(discovered_repos)
            async for repo_info in self._search_repositories(query, language, remaining_count):
                if repo_info.url not in exclude_repos:
                    discovered_repos.append(repo_info)
                    if len(discovered_repos) >= max_repos:
                        break

        logger.info(
            "Repository discovery completed",
            total_discovered=len(discovered_repos),
            failed_count=len(self._failed_repos),
        )

        return discovered_repos

    async def _process_included_repos(self, repo_urls: List[str]) -> AsyncGenerator[RepositoryInfo, None]:
        """Process explicitly included repositories."""
        tasks = []
        for url in repo_urls:
            task = asyncio.create_task(self._get_repository_info(url))
            tasks.append(task)

        for task in asyncio.as_completed(tasks):
            try:
                repo_info = await task
                if repo_info:
                    yield repo_info
            except Exception as e:
                logger.error("Failed to process included repository", error=str(e))

    async def _search_repositories(
        self, query: str, language: str, max_count: int
    ) -> AsyncGenerator[RepositoryInfo, None]:
        """Search for repositories using GitHub API."""
        if not self.github:
            return

        search_query = f"{query} language:{language}"
        logger.info("Searching GitHub repositories", query=search_query)

        try:
            repositories = self.github.search_repositories(query=search_query, sort="stars", order="desc")

            count = 0
            for repo in repositories:
                if count >= max_count:
                    break

                try:
                    repo_info = await self._convert_github_repo_to_info(repo)
                    if repo_info:
                        yield repo_info
                        count += 1

                        # Rate limiting
                        if count % 10 == 0:
                            await asyncio.sleep(1)

                except RateLimitExceededException:
                    logger.warning("GitHub rate limit exceeded, waiting...")
                    await self._handle_rate_limit()
                    break
                except Exception as e:
                    logger.error("Error processing repository", repo=repo.full_name, error=str(e))
                    continue

        except Exception as e:
            logger.error("Error searching repositories", error=str(e))

    async def _get_repository_info(self, url: str) -> Optional[RepositoryInfo]:
        """Get repository information from URL."""
        try:
            if not self.github:
                return None

            # Parse repository from URL
            parsed = urlparse(url)
            repo_path = parsed.path.strip("/")

            async with self.throttler:
                repo = self.github.get_repo(repo_path)
                return await self._convert_github_repo_to_info(repo)

        except Exception as e:
            logger.error("Failed to get repository info", url=url, error=str(e))
            self._failed_repos[url] = str(e)
            return None

    async def _convert_github_repo_to_info(self, repo: Repository) -> Optional[RepositoryInfo]:
        """Convert GitHub repository to RepositoryInfo."""
        try:
            # Validate repository
            if not await self._validate_repository(repo):
                return None

            return RepositoryInfo(
                url=repo.html_url,
                name=repo.name,
                full_name=repo.full_name,
                description=repo.description or "",
                language=repo.language or "Unknown",
                stars=repo.stargazers_count,
                forks=repo.forks_count,
                created_at=repo.created_at,
                updated_at=repo.updated_at,
                license=repo.license.spdx_id if repo.license else None,
                default_branch=repo.default_branch,
                is_fork=repo.fork,
                size=repo.size,
                status=RepositoryStatus.DISCOVERED,
            )
        except Exception as e:
            logger.error("Error converting repository", repo=repo.full_name, error=str(e))
            return None

    async def _validate_repository(self, repo: Repository) -> bool:
        """Validate if repository meets criteria."""
        try:
            # Check if repository is blocked
            if not self.config.validate_repository_url(repo.html_url):
                logger.debug("Repository is blocked", repo=repo.full_name)
                return False

            # Check if it's a fork (if we want to exclude forks)
            if repo.fork:
                logger.debug("Repository is a fork", repo=repo.full_name)
                return False

            # Check license restrictions
            if self.config.security.allowed_licenses:
                repo_license = repo.license.spdx_id if repo.license else None
                if repo_license not in self.config.security.allowed_licenses:
                    logger.debug("Repository license not allowed", repo=repo.full_name, license=repo_license)
                    return False

            # Check for Lean4 files
            contents = repo.get_contents("")
            content_names = [content.name for content in contents]

            # Must have lakefile.lean
            if "lakefile.lean" not in content_names:
                logger.debug("Repository missing lakefile.lean", repo=repo.full_name)
                return False

            # Must have lean-toolchain
            if "lean-toolchain" not in content_names:
                logger.debug("Repository missing lean-toolchain", repo=repo.full_name)
                return False

            # Must not be Lean3 (no leanpkg.toml)
            if "leanpkg.toml" in content_names:
                logger.debug("Repository is Lean3", repo=repo.full_name)
                return False

            return True

        except Exception as e:
            logger.error("Error validating repository", repo=repo.full_name, error=str(e))
            return False

    async def clone_repositories(
        self, repositories: List[RepositoryInfo], force_update: bool = False
    ) -> List[RepositoryInfo]:
        """Clone repositories asynchronously."""
        logger.info("Starting repository cloning", count=len(repositories))

        # Create tasks for concurrent cloning
        tasks = []
        for repo in repositories:
            task = asyncio.create_task(self._clone_single_repository(repo, force_update))
            tasks.append(task)

        # Process tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.config.github.max_concurrent_requests)

        async def limited_clone(task):
            async with semaphore:
                return await task

        cloned_repos = []
        for task in asyncio.as_completed([limited_clone(t) for t in tasks]):
            try:
                repo_info = await task
                if repo_info and repo_info.status == RepositoryStatus.CLONED:
                    cloned_repos.append(repo_info)
            except Exception as e:
                logger.error("Error in repository cloning", error=str(e))

        logger.info("Repository cloning completed", cloned_count=len(cloned_repos))
        return cloned_repos

    async def _clone_single_repository(
        self, repo_info: RepositoryInfo, force_update: bool = False
    ) -> Optional[RepositoryInfo]:
        """Clone a single repository."""
        repo_path = self.config.repos_dir / repo_info.full_name.replace("/", "_")

        try:
            if repo_path.exists() and not force_update:
                # Repository already exists, check if it needs updating
                if await self._needs_update(repo_path, repo_info):
                    await self._update_repository(repo_path, repo_info)
                else:
                    logger.debug("Repository up to date", repo=repo_info.full_name)
                    repo_info.status = RepositoryStatus.CLONED
                    return repo_info
            else:
                # Clone new repository
                await self._perform_clone(repo_path, repo_info)

            repo_info.status = RepositoryStatus.CLONED
            repo_info.local_path = repo_path
            return repo_info

        except Exception as e:
            logger.error(
                "Failed to clone repository",
                repo=repo_info.full_name,
                error=str(e),
            )
            repo_info.status = RepositoryStatus.FAILED
            repo_info.error_message = str(e)
            return repo_info

    async def _perform_clone(self, repo_path: Path, repo_info: RepositoryInfo) -> None:
        """Perform the actual cloning operation."""
        async with self.throttler:
            # Use dulwich for async git operations
            porcelain.clone(
                repo_info.url,
                target=str(repo_path),
                checkout=True,
                depth=1,  # Shallow clone for speed
            )

        logger.info("Repository cloned successfully", repo=repo_info.full_name)

    async def _needs_update(self, repo_path: Path, repo_info: RepositoryInfo) -> bool:
        """Check if repository needs updating."""
        try:
            repo = Repo(str(repo_path))
            head = repo.head()
            if head:
                current_commit = head.decode("utf-8")
                # Compare with latest commit from GitHub
                return current_commit != repo_info.latest_commit
        except Exception as e:
            logger.warning("Error checking repository update", repo=repo_info.full_name, error=str(e))

        return True

    async def _update_repository(self, repo_path: Path, repo_info: RepositoryInfo) -> None:
        """Update existing repository."""
        try:
            async with self.throttler:
                repo = Repo(str(repo_path))
                porcelain.pull(repo, repo_info.url)

            logger.info("Repository updated successfully", repo=repo_info.full_name)
        except Exception as e:
            logger.error("Failed to update repository", repo=repo_info.full_name, error=str(e))
            raise

    async def _handle_rate_limit(self) -> None:
        """Handle GitHub rate limiting."""
        if self.github:
            rate_limit = self.github.get_rate_limit()
            reset_time = rate_limit.core.reset
            wait_time = (reset_time - datetime.now()).total_seconds()

            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time} seconds")
                await asyncio.sleep(wait_time + 1)

    def get_statistics(self) -> Dict[str, Any]:
        """Get repository processing statistics."""
        return {
            "processed_count": len(self._processed_repos),
            "failed_count": len(self._failed_repos),
            "failed_repos": self._failed_repos,
            "rate_limit_reset": self._rate_limit_reset,
        }
