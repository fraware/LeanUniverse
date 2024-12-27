# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import getpass
import os
from dataclasses import dataclass, field
from pathlib import Path

from psutil import cpu_count

from lean_universe.utils.params import Params

# from logging import getLogger


# logger = getLogger()

# @cache
# def get_git_repo_root() -> str:
#     """
#     Returns the top-level directory of the current Git repository.
#     Returns:
#         str: The absolute path of the top-level directory of the Git repository.
#              Returns None if the current directory is not part of a Git repository.
#     """

#     try:
#         # Run the git command to get the top-level directory
#         repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()  # nosec
#         return repo_root
#     except subprocess.CalledProcessError:
#         # Handle the case where the current directory is not part of a Git repo
#         logger.error("Current directory is not a Git repository.")
#         return ""

USER = getpass.getuser()

# ROOT_WORKING_DIR = Path(get_git_repo_root())
# if not Path(""):
ROOT_WORKING_DIR = Path(__file__).parent.parent.parent
# else:
#     ROOT_WORKING_DIR = Path(ROOT_WORKING_DIR)

cache_path = os.getenv("LEANUNIVERSE_CACHE", None)
if cache_path:
    CACHE_DIR = Path(f"{cache_path}")
else:
    CACHE_DIR = ROOT_WORKING_DIR / "cache"

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

LEAN3_REPO = "leanpkg.toml"
LEAN4_LAKEFILE = "lakefile.lean"
LEAN4_TOOLCHAIN = "lean-toolchain"

LARGE_DATASET = 100000
LARGE_DATASET_TEST_VAL_PERCENT = 2
MEDIUM_DATASET = 1000
MEDIUM_DATASET_TEST_VAL_PERCENT = 7
SMALL_DATASET = 100
SMALL_DATASET_TEST_VAL_PERCENT = 10
TEST_VAL_MIN_SIZE = 5


@dataclass
class EvalArgs(Params):
    cache_dir: str = ""
    working_dir: str = str(ROOT_WORKING_DIR)
    dataset_export_dir: str = ""
    raw_dataset_dir: str = ""  # raw files to compose the dataset
    repos_dir: str = ""  # cloned repos
    repos_included: list = field(default_factory=list)  # repos to include in the dataset
    max_num_repos: int = 1
    ld_max_num_procs: int = 32
    large_dataset: int = LARGE_DATASET
    large_dataset_test_val_percent: int = LARGE_DATASET_TEST_VAL_PERCENT
    medium_dataset: int = MEDIUM_DATASET
    medium_dataset_test_val_percent: int = MEDIUM_DATASET_TEST_VAL_PERCENT
    small_dataset: int = SMALL_DATASET
    small_dataset_test_val_percent: int = SMALL_DATASET_TEST_VAL_PERCENT
    test_val_min_size: int = TEST_VAL_MIN_SIZE
    # dependencies_build_dir: str = str(DEPENDENCIES_BUILD_DIR)
    # dependencies_bashrc_extension: str = str(BASHRC_EXTENSION_FILE)
    log_file: str = ""
    timestamp: str = TIMESTAMP
    # log_file_dependencies: str = str(LOG_FILE_DEPENDENCIES)
    # log_failed_repos: str = str(LOG_FAILED_REPOS)
    # lean_extractor_file: str = str(LEAN4_DATA_EXTRACTOR_PATH)
    # lean_connector_file: str = str(LEAN4_DATA_CONNECTOR_PATH)
    num_threads: int = cpu_count(logical=False)

    def __post_init__(self):
        if self.cache_dir == "":
            self.cache_dir = CACHE_DIR
        else:
            self.cache_dir = Path(self.cache_dir)
        self.cache_dir = Path(self.cache_dir)
        if self.log_file == "":
            self.log_file = self.cache_dir / f"logs/{TIMESTAMP}_{USER}_lean_universe.log"
        else:
            self.log_file = Path(self.log_file)
        if self.dataset_export_dir == "":
            self.dataset_export_dir = self.cache_dir / "dataset"
        else:
            self.dataset_export_dir = Path(self.dataset_export_dir)
        if self.raw_dataset_dir == "":
            self.raw_dataset_dir = self.cache_dir / "raw"
        else:
            self.raw_dataset_dir = Path(self.raw_dataset_dir)
        if self.repos_dir == "":
            self.repos_dir = self.cache_dir / "repos"
        else:
            self.repos_dir = Path(self.repos_dir)
        self.log_file = Path(self.log_file)
        self.working_dir = Path(self.working_dir)
        self.dataset_export_dir = Path(self.dataset_export_dir)
        self.raw_dataset_dir = Path(self.raw_dataset_dir)
        self.repos_dir = Path(self.repos_dir)
        self.timestamp = TIMESTAMP
        # self.log_failed_repos = Path(self.log_failed_repos)

        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dataset_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_export_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "logs").mkdir(parents=True, exist_ok=True)

        if not self.log_file.is_file():
            self.log_file.touch()

        # if not self.log_failed_repos.is_file():
        #     self.log_failed_repos.touch()
