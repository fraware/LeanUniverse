# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from functools import cache
from logging import getLogger
from pathlib import Path
from subprocess import PIPE, CalledProcessError, Popen  # nosec

from github import Auth, Github
from github.Repository import Repository

logger = getLogger()


# @cache
def get_github() -> Github:
    """
    Returns a Github object with authentication if the GITHUB_ACCESS_TOKEN
    environment variable is set.
    """

    # This provides increased rate limit for GitHub API calls.
    access_token = os.getenv("GITHUB_ACCESS_TOKEN", None)
    if access_token:
        logger.info("Using GitHub personal access token for authentication")
        github = Github(auth=Auth.Token(access_token))
        github.get_user().login
        return github
    else:
        logger.info("Using GitHub without authentication. Don't be surprised if you hit the API rate limit.")
        return Github()


def normalize_url(url: str) -> str:
    """
    Normalize the given URL by removing any trailing slashes.
    Args:
        url (str): The URL to be normalized.
    Returns:
        str: The normalized URL.
    """

    return url.rstrip("/")


@cache
def url_to_repo(GITHUB: Github, url: str, num_retries: int = 2) -> Repository:
    """
    Retrieves a GitHub repository object based on the given URL.
    Args:
        url (str): The URL of the repository.
        num_retries (int, optional): The number of retries in case of failure. Defaults to 2.
    Returns:
        Repository: The GitHub repository object.
    Raises:
        Exception: If the retrieval fails after the specified number of retries.
    """
    logger.debug(f"Fetching repository for {url}")
    url = normalize_url(url)
    backoff = 1

    while True:
        try:
            return GITHUB.get_repo("/".join(url.split("/")[-2:]))
        except Exception as ex:
            if num_retries <= 0:
                logger.error(f"url_to_repo({url}) failed after {num_retries} retries.")
                raise ex
            num_retries -= 1
            logger.debug(f'url_to_repo("{url}") failed. Retrying...')
            time.sleep(backoff)
            backoff


@contextmanager
def change_directory(new_dir: str) -> Generator[None, None, None]:
    """
    Change the current working directory to the specified directory temporarily.
    Args:
        new_dir (str): The path of the directory to change to.
    Yields:
        None
    Raises:
        OSError: If the specified directory does not exist or cannot be accessed.
    Example:
        >>> with change_directory('/path/to/new_directory'):
        ...     # Code to be executed in the new directory
        ...
    """

    current_dir = Path.cwd()
    try:
        # Change to the new directory
        os.chdir(new_dir)
        yield
    finally:
        # Restore the original directory
        os.chdir(current_dir)


def execute_and_capture(command: str, directory: Path) -> tuple[str, str]:
    """
    Executes a shell command in the specified directory and captures the stdout and stderr outputs.
    Args:
        command (str): The shell command to execute.
        directory (Path): The directory in which to execute the command.
    Returns:
        tuple[str, str]: A tuple containing the stdout and stderr outputs of the command.
    """

    # Use the custom context manager to handle directory changes
    with change_directory(directory.as_posix()), Popen(
        command, shell=True, stdout=PIPE, stderr=PIPE, text=True  # nosec
    ) as process:
        if process.stdout:
            output = process.stdout.read().strip()
            for line in output.splitlines():
                logger.info(line.strip())
        process.wait()
        if process.returncode != 0 and process.stderr:
            error_message = process.stderr.read().strip()
            logger.error(f"Command '{command}' failed with exit code {error_message}.")
            raise CalledProcessError(process.returncode, command, error_message)
        return output, process.stderr.read().strip() if process.stderr else ""


def clone_and_checkout(repo_url: str, repo_prefix: str, repo_commit: str, dir: Path, exept: bool = True) -> None:
    logger.info(f"{repo_prefix} Cloning repo to {dir.as_posix()}.")
    clone_command = f"git clone -n --recursive {repo_url} {dir.name}"
    try:
        execute_and_capture(clone_command, dir.parent)
    except CalledProcessError as ex:
        logger.error(f"{repo_prefix} Failed to clone.")
        if exept:
            raise ex
    checkout_command = f"git checkout {repo_commit} && git submodule update --recursive"
    try:
        execute_and_capture(checkout_command, dir)
    except CalledProcessError as ex:
        logger.error(f"{repo_prefix} Failed to checkout at {repo_commit}.")
        if exept:
            raise ex
    logger.info(f"{repo_prefix} Cloned to {dir}")


def reset_and_pull(repo_url: str, repo_prefix: str, repo_commit: str, dir: Path, exept: bool = True) -> None:
    logger.info(f"{repo_prefix} Resetting and pulling repo at {dir.as_posix()}.")
    pull_command = f"git checkout {repo_commit} && git pull && git submodule update --recursive"
    try:
        execute_and_capture(pull_command, dir)
    except CalledProcessError as ex:
        logger.error(f"{repo_prefix} Failed to clone.")
        if exept:
            raise ex
    logger.info(f"{repo_prefix} Resetted and pulled. {dir}")
