# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import shutil
import time
from collections import defaultdict
from copy import copy
from datetime import datetime
from logging import getLogger
from pathlib import Path
from subprocess import CalledProcessError  # nosec
from typing import Optional, Union

import lean_dojo
import networkx as nx
from github import Github, RateLimitExceededException, Repository
from lean_dojo import LeanGitRepo, TracedRepo, TracedTheorem, constants, trace
from lean_dojo.constants import LEAN4_PACKAGES_DIR
from tqdm import tqdm

from lean_universe.dataset.args import LEAN3_REPO, LEAN4_LAKEFILE, LEAN4_TOOLCHAIN, USER
from lean_universe.utils.tools import (
    clone_and_checkout,
    execute_and_capture,
    get_github,
    reset_and_pull,
    url_to_repo,
)

logger = getLogger()

SPLIT_NAME = str  # train/val/test
SPLIT = dict[SPLIT_NAME, list[TracedTheorem]]
SPLIT_STRATEGY = str

random.seed(3407)


def get_lean_repos(
    GITHUB: Github,
    repos_included: Optional[list[str]] = None,
    max_num: int = -1,
    query: str = "lean",
    language: str = "lean",
):
    """
    Retrieves Lean repositories from GitHub based on the specified query and language.
    Args:
        repos_included (Optional[list[str]], optional): A list of repository names to include. Defaults to None.
        max_num (int, optional): The maximum number of repositories to retrieve. Defaults to None.
        query (str, optional): The search query. Defaults to "lean".
        language (str, optional): The programming language. Defaults to "lean".
    Returns:
        List[Tuple[str, str]]: A list of tuples containing the full name and HTML URL of the repositories.
    """

    logger.info("Searching for Lean repositories on GitHub...")
    query = f"{query} language:{language}"
    results = []
    if repos_included:
        for repo_name in repos_included:
            repo = url_to_repo(GITHUB=GITHUB, url=repo_name)
            if repo:
                results.append((repo.full_name, repo.html_url))
            if max_num > 0 and len(results) >= max_num:
                break

    logger.info(f"Found {len(results)} repositories in the included list.")
    if max_num > 0 and len(results) >= max_num:
        return results
    else:
        max_num = max_num - len(results) if max_num > 0 else max_num

    repositories = GITHUB.search_repositories(query=query, sort="stars", order="desc")
    try:
        for repo in tqdm(repositories):
            results.append((repo.full_name, repo.html_url))
            if max_num > 0 and len(results) >= max_num:
                break
            if len(results) % 1000 == 0:
                logger.info(f"Fetched {len(results)} results so far...")
                time.sleep(10)  # Sleep for 10 seconds every 1000 results to manage rate limits
    except RateLimitExceededException:
        print("GitHub API rate limit exceeded. Please wait and try again later.")
    return results


class Dataset:
    def __init__(
        self,
        cache_dir: Path,
        log_file: Path,
        dataset_export_dir: Path,
        raw_dataset_dir: Path,
        repos_dir: Path,
        timestamp: str,
        github: Github,
        ld_cache_dir: Optional[Path],
        repos_excluded: Optional[list[str]] = None,
        repos_included: Optional[list[str]] = None,
        licesnes_excluded: Optional[list[str]] = None,
        ld_max_num_procs: int = -1,
        large_dataset: int = -1,
        large_dataset_test_val_percent: int = -1,
        medium_dataset: int = -1,
        medium_dataset_test_val_percent: int = -1,
        small_dataset: int = -1,
        small_dataset_test_val_percent: int = -1,
        test_val_min_size: int = -1,
    ):
        self.cache_dir = cache_dir
        self.database_file = cache_dir / "database.json"

        self.dataset_export_dir = dataset_export_dir
        self.raw_dataset_dir = raw_dataset_dir
        self.repos_dir = repos_dir
        self.timestamp = timestamp
        self.repos_excluded = repos_excluded
        self.repos_included = repos_included
        self.licenses_excluded = licesnes_excluded
        self.log_file = log_file
        self.ld_max_num_procs = ld_max_num_procs
        self.ld_cache_dir = ld_cache_dir

        self.large_dataset = large_dataset
        self.large_dataset_test_val_percent = large_dataset_test_val_percent
        self.medium_dataset = medium_dataset
        self.medium_dataset_test_val_percent = medium_dataset_test_val_percent
        self.small_dataset = small_dataset
        self.small_dataset_test_val_percent = small_dataset_test_val_percent

        self.github = github

        self.database: dict[str, dict] = {}
        self.database["report"] = {}
        self.database["repos"] = {}
        self.database["cache_repos"] = {}
        self.database["cache_report"] = {}
        if self.database_file.is_file():
            logger.info(f"Loading database file from {self.database_file}")
            with self.database_file.open("r", encoding="utf-8") as file:
                self.database = json.load(file)
            for repo in self.database["repos"]:
                if self.database["repos"][repo]["is_correct"]:
                    self.database["cache_repos"][repo] = self.database["repos"][repo]
            self.database["repos"] = {}
            self.database["cache_report"] = self.database["report"]
            self.database["report"] = {}

        self.database["report"]["cache_dir"] = self.cache_dir.as_posix()
        self.database["report"]["log_file"] = self.log_file.as_posix()
        self.database["report"]["raw_dataset_dir"] = self.raw_dataset_dir.as_posix()
        self.database["report"]["repos_dir"] = self.repos_dir.as_posix()
        self.database["report"]["dataset_export_dir"] = self.dataset_export_dir.as_posix()
        self.database["report"]["timestamp"] = self.timestamp
        self.database["report"]["user"] = USER
        self.database["report"]["repos_excluded"] = self.repos_excluded
        self.database["report"]["repos_included"] = self.repos_included
        self.database["report"]["licenses_excluded"] = self.licenses_excluded

        self.new_repos: set[str] = set()
        self.updated_repos: set[str] = set()
        self.lean_ready_repos: set[str] = set()

    def report(self) -> None:
        logger.info(f"Saving database file to {self.database_file}.")
        with open(self.database_file, "w", encoding="utf-8") as file:
            json.dump(self.database, file, indent=4)

    def __get_repo_info(self, repo: Repository) -> tuple[dict, bool]:
        info = {}
        print_prefix = f"[{repo.full_name}]"

        contents = repo.get_contents("")
        contents_names = [content_file.name for content_file in contents]

        info["date_updated"] = repo.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        info["date_created"] = repo.created_at.strftime("%Y-%m-%d %H:%M:%S")
        info["full_name"] = repo.full_name
        info["name"] = repo.name
        info["description"] = repo.description
        info["branch"] = repo.default_branch
        info["sha"] = repo.get_branch(repo.default_branch).commit.sha
        info["star_count"] = repo.stargazers_count
        info["fork_count"] = repo.forks_count
        info["license"] = repo.license.spdx_id if repo.license else "None"
        info["path"] = (self.repos_dir / repo.full_name.replace("/", "_")).as_posix()
        info["raw_dataset_path"] = (self.raw_dataset_dir / repo.full_name.replace("/", "_") / info["sha"]).as_posix()
        info["dataset_export_dir"] = (
            self.dataset_export_dir / repo.full_name.replace("/", "_") / info["sha"]
        ).as_posix()
        info["is_correct"] = False

        if repo.fork:
            info["is_fork"] = True
            logger.warning(f"{print_prefix} Repo is a fork.")
            return (info, False)

        if LEAN3_REPO in contents_names:
            info["lean3_repo"] = True
            logger.warning(f"{print_prefix} Repo is a Lean3 repo.")
            return (info, False)

        if LEAN4_LAKEFILE not in contents_names:
            info["lean4_lakefile_missing"] = True
            logger.warning(f"{print_prefix} Repo does not have a lakefile.")
            return (info, False)

        if LEAN4_TOOLCHAIN not in contents_names:
            info["lean4_toolchain_missing"] = True
            logger.warning(f"{print_prefix} Repodoes not have a lean-toolchain.")
            return (info, False)

        # Get the Lean version
        toolchain = repo.get_contents(LEAN4_TOOLCHAIN)
        info["toolchain"] = toolchain.decoded_content.decode("utf-8")
        info["is_correct"] = True
        return (info, True)

    def __license_excluded(self, license: str) -> bool:
        if self.licenses_excluded:
            for license_exclude in self.licenses_excluded:
                if license_exclude in license:
                    return True
        return False

    def filter_new_repos(self, repos: list) -> None:
        self.database["report"]["repos_found"] = len(repos)
        self.database["report"]["repos_incorrect_lean"] = 0
        self.database["report"]["repos_incorrect_exclusion"] = 0
        self.database["report"]["repos_incorrect_license"] = 0
        self.database["report"]["repos_correct"] = 0

        for repo in tqdm(repos):
            url = repo[1]
            rep: Repository = url_to_repo(self.github, repo[1])
            print_prefix = f"[{rep.full_name}]"
            logger.info(f"{print_prefix} Processing ...")

            info, correct = self.__get_repo_info(rep)
            if not correct:
                self.database["report"]["repos_incorrect_lean"] += 1
                self.database["repos"][url] = info
                logger.info(f"{print_prefix} is not a Lean4 repo.")
                continue

            # Filter out repositories with excluded licenses
            if self.__license_excluded(info["license"]):
                info["Excluded_license"] = True
                info["is_correct"] = False
                self.database["report"]["repos_incorrect_license"] += 1
                self.database["repos"][url] = info
                logger.info(f"{print_prefix} has an excluded license. Skipping...")
                continue

            # Filter out repositories that are in the exclude list
            if self.repos_excluded and url in self.repos_excluded:
                info["Excluded_repo"] = True
                info["is_correct"] = False
                self.database["report"]["repos_incorrect_exclusion"] += 1
                self.database["repos"][url] = info
                logger.info(f"{print_prefix} is in the exclude list. Skipping...")
                continue

            self.database["repos"][url] = info
            self.database["report"]["repos_correct"] += 1
            self.new_repos.add(url)

        correct = self.database["report"]["repos_correct"]
        found = self.database["report"]["repos_found"]
        logger.info(f"Using {correct}/{found} repositories.")

    def clone_or_pull_repos(self) -> None:
        for repo in tqdm(self.new_repos):
            repo_info = self.database["repos"][repo]
            print_prefix = f"[{repo_info['full_name']}]"
            if repo in self.database["cache_repos"]:
                if self.database["cache_repos"][repo]["sha"] != repo_info["sha"]:
                    if Path(repo_info["path"]).is_dir():
                        reset_and_pull(
                            repo, f"""[{repo_info["full_name"]}]""", repo_info["branch"], Path(repo_info["path"])
                        )
                        logger.info(f"{print_prefix} Repository has been updated. Resetting and pulling...")
                    else:
                        clone_and_checkout(
                            repo, f"""[{repo_info["full_name"]}]""", repo_info["branch"], Path(repo_info["path"])
                        )
                        logger.info(f"{print_prefix} Repository has been updated. Cloning...")
                    self.updated_repos.add(repo)
                else:
                    if not Path(repo_info["path"]).is_dir():
                        clone_and_checkout(
                            repo, f"""[{repo_info["full_name"]}]""", repo_info["branch"], Path(repo_info["path"])
                        )
                        logger.info(f"{print_prefix} Repository has been updated. Cloning...")
                        self.updated_repos.add(repo)

            else:
                if Path(repo_info["path"]).is_dir():
                    reset_and_pull(
                        repo, f"""[{repo_info["full_name"]}]""", repo_info["branch"], Path(repo_info["path"])
                    )
                else:
                    clone_and_checkout(
                        repo, f"""[{repo_info["full_name"]}]""", repo_info["branch"], Path(repo_info["path"])
                    )
                self.updated_repos.add(repo)
        self.database["report"]["repos_correct_updated"] = len(self.updated_repos)
        logger.info(f"""[Clone or Fetch] Updated {self.database["report"]["repos_correct_updated"]} repositories.""")

    def __check_for_incorrect_datasets(self) -> None:
        for repo in self.new_repos:
            if not Path(self.database["repos"][repo]["raw_dataset_path"]).is_dir():
                self.updated_repos.add(repo)

    def __build_package(self, dir: Path, name: str) -> bool:
        try:
            logger.info(f"[{name}] Get caches.")
            command = "lake exe cache get"
            execute_and_capture(command, dir)
        except CalledProcessError as ex:
            logger.warning(f"[{name}] Failed to get caches. {ex.stderr}")

            # return False
            # raise ex

        try:
            logger.info(f"[{name}] Building package.")
            command = "lake build"
            execute_and_capture(command, dir)
        except CalledProcessError as ex:
            logger.error(f"[{name}] Failed to build package. {ex.stderr}")
            return False
            # raise ex

        try:
            logger.info(f"[{name}] Upgrading package.")
            command = "lake update"
            execute_and_capture(command, dir)
        except CalledProcessError as ex:
            logger.error(f"[{name}] Failed to upgrade package. {ex.stderr}")
            return False
            # raise ex

        try:
            logger.info(f"[{name}] Building package.")
            command = "lake build"
            execute_and_capture(command, dir)
        except CalledProcessError as ex:
            logger.error(f"[{name}] Failed to build package. {ex.stderr}")
            return False
            # raise ex

        logger.info(f"[{name}] Package built and updated successfully.")
        return True

    def build_lake(self) -> None:
        self.__check_for_incorrect_datasets()
        for repo in tqdm(self.updated_repos):
            repo_info = self.database["repos"][repo]
            if self.__build_package(Path(repo_info["path"]), repo_info["full_name"]):
                self.database["repos"][repo]["builds"] = True
                self.lean_ready_repos.add(repo)
            else:
                self.database["repos"][repo]["builds"] = False

    def configure_leandojo(self) -> None:
        if self.ld_max_num_procs > 0:
            constants.MAX_NUM_PROCS = self.ld_max_num_procs
            logger.info(f"[LeanDojo] Using {constants.MAX_NUM_PROCS} processes.")
        if self.ld_cache_dir:
            constants.CACHE_DIR = self.ld_cache_dir
            logger.info(f"[LeanDojo] Using cache directory {constants.CACHE_DIR}")

        self.database["report"]["LeanDojo"] = {}
        self.database["report"]["LeanDojo"]["version"] = constants.__version__
        self.database["report"]["LeanDojo"]["MAX_NUM_PROCS"] = constants.MAX_NUM_PROCS
        self.database["report"]["LeanDojo"]["CACHE_DIR"] = constants.CACHE_DIR.as_posix()

    def run_leandojo(self) -> None:
        logger.info("Running LeanDojo for the data extraction.")
        self.configure_leandojo()
        for repo in tqdm(self.lean_ready_repos if self.lean_ready_repos else self.new_repos):
            repo_info = self.database["repos"][repo]
            print_prefix = f"[{repo_info['full_name']}]"
            logger.info(f"{print_prefix} Extracting dataset.")

            try:
                lean_repo = LeanGitRepo(repo_info["path"], repo_info["branch"])
                traced_repo = trace(
                    repo=lean_repo,
                    dst_dir=repo_info["raw_dataset_path"],
                    # build_deps=True
                )
                self.database["repos"][repo]["Traced"] = True
                splits = self.split_data(traced_repo, repo)
                self.export_data(
                    traced_repo,
                    splits,
                    repo_info["dataset_export_dir"],
                    dataset_name=f"{print_prefix} dataset",
                    repo=repo,
                )
            except Exception as ex:
                logger.info(f"{print_prefix} Error fetching results: {repr(ex)}")

    def split_data(self, traced_repo: TracedRepo, repo: str) -> dict[SPLIT_STRATEGY, SPLIT]:
        # Skip theorems in the Lean 4 repo itself.
        traced_theorems = [thm for thm in traced_repo.get_traced_theorems() if not thm.repo.is_lean4]
        repo_info = self.database["repos"][repo]
        print_prefix = f"[{repo_info['full_name']}]"

        total_theorems_num = len(traced_theorems)
        logger.info(f"{print_prefix} Theorems in total: {total_theorems_num}")

        NUM_VAL = NUM_TEST = 0
        if total_theorems_num > self.large_dataset:
            NUM_VAL = NUM_TEST = int(total_theorems_num * self.large_dataset_test_val_percent / 100)
        else:
            if total_theorems_num > self.medium_dataset:
                NUM_VAL = NUM_TEST = int(total_theorems_num * self.medium_dataset_test_val_percent / 100)
            else:
                if total_theorems_num > self.small_dataset:
                    NUM_VAL = NUM_TEST = int(total_theorems_num * self.small_dataset_test_val_percent / 100)

        self.database["repos"][repo][
            "Split train/val/test"
        ] = f"{total_theorems_num - NUM_TEST - NUM_VAL}/{NUM_VAL}/{NUM_TEST}"
        return {
            "random": self.split_randomly(traced_theorems, NUM_TEST, NUM_VAL, repo),
            "novel_premises": self.split_by_premise(traced_theorems, NUM_TEST, NUM_VAL, repo),
        }

    def split_randomly(self, traced_theorems: list[TracedTheorem], NUM_TEST: int, NUM_VAL: int, repo: str) -> SPLIT:
        """Split ``traced_theorems`` randomly into train/val/test."""
        repo_info = self.database["repos"][repo]
        print_prefix = f"[{repo_info['full_name']}]"
        logger.info(f"{print_prefix} Splitting the theorems randomly")
        traced_theorems = copy(traced_theorems)
        random.shuffle(traced_theorems)
        return self._split_sequentially(traced_theorems, NUM_TEST, NUM_VAL)

    def _split_sequentially(self, traced_theorems: list[TracedTheorem], NUM_TEST: int, NUM_VAL: int) -> SPLIT:
        """Split ``traced_theorems`` sequentially into train/val/test."""
        num_theorems = len(traced_theorems)
        num_train = num_theorems - NUM_VAL - NUM_TEST
        return {
            "train": traced_theorems[:num_train],
            "val": traced_theorems[num_train : num_train + NUM_VAL],
            "test": traced_theorems[num_train + NUM_VAL :],
        }

    def split_by_premise(self, traced_theorems: list[TracedTheorem], NUM_TEST: int, NUM_VAL: int, repo: str) -> SPLIT:
        """
        Split theorems into train/val/test so that proofs in val/test rely on at
        least one novel premise that does not appear in train.
        """
        repo_info = self.database["repos"][repo]
        print_prefix = f"[{repo_info['full_name']}]"
        logger.info(f"{print_prefix} Splitting the theorems by premises")

        # Figure out the number of theorems in train/val/test.
        # num_theorems = len(traced_theorems)
        num_val_test = NUM_VAL + NUM_TEST
        # num_train = num_theorems - num_val_test
        theorems_val_test_: set = set()

        # Map each premise to a list of theorems using it.
        theorems_by_premises_ = defaultdict(list)
        for t in traced_theorems:
            for p in t.get_premise_full_names():
                theorems_by_premises_[p].append(t)

        # Sort the premises by the number of theorems using them (in ascending order).
        theorems_by_premises = sorted(theorems_by_premises_.items(), key=lambda x: len(x[1]))

        # For each premise, put all theorems using it into val_test so that it does not appear in train.
        for _, thms in theorems_by_premises:
            if len(theorems_val_test_) < num_val_test:
                theorems_val_test_.update(thms)

        # All other theorems go to train.
        theorems_train = [t for t in traced_theorems if t not in theorems_val_test_]
        theorems_val_test = list(theorems_val_test_)
        random.shuffle(theorems_val_test)

        return {
            "train": theorems_train,
            "val": theorems_val_test[:NUM_VAL],
            "test": theorems_val_test[NUM_VAL:],
        }

    def export_data(
        self,
        traced_repo: TracedRepo,
        splits: dict[SPLIT_STRATEGY, SPLIT],
        dst_path: Union[str, Path],
        repo: str,
        **kwargs,
    ) -> None:
        """Export a traced repo whose theorems have been splitted to ``dst_path``."""
        repo_info = self.database["repos"][repo]
        print_prefix = f"[{repo_info['full_name']}]"
        if isinstance(dst_path, str):
            dst_path = Path(dst_path)
        if dst_path.exists():
            logger.warning(f"{dst_path} already exists. Removing it now.")
            shutil.rmtree(dst_path)

        # Export the proofs.
        logger.info(f"{print_prefix} Exporting the proofs.")
        self.export_proofs(splits, dst_path, traced_repo, repo)

        # Export the premises (theorems, definitions, etc.).
        logger.info(f"{print_prefix} Exporting the premises.")
        self.export_premises(traced_repo, dst_path, repo)

        # Export the licenses.
        logger.info(f"{print_prefix} Exporting the licenses.")
        self.export_licenses(traced_repo, dst_path)

        # Export metadata.
        logger.info(f"{print_prefix} Exporting the metadata.")
        self.export_metadata(traced_repo, dst_path, **kwargs)

        logger.info(f"{print_prefix} Exporting the data is completed.")

    def export_proofs(
        self, splits: dict[SPLIT_STRATEGY, SPLIT], dst_path: Path, traced_repo: TracedRepo, repo: str
    ) -> None:
        """Export all proofs in a traced repo to ``dst_path''."""
        repo_info = self.database["repos"][repo]
        print_prefix = f"[{repo_info['full_name']}]"
        for strategy, split in splits.items():
            split_dir = dst_path / strategy
            split_dir.mkdir(parents=True)

            for name, theorems in split.items():
                data = []
                num_tactics = 0

                for thm in theorems:
                    tactics = [
                        {
                            "tactic": t.tactic,
                            "annotated_tactic": t.get_annotated_tactic(),
                            "state_before": t.state_before,
                            "state_after": t.state_after,
                        }
                        for t in thm.get_traced_tactics()
                        if t.state_before != "no goals" and "·" not in t.tactic  # Ignore "·".
                    ]
                    num_tactics += len(tactics)
                    data.append(
                        {
                            "url": traced_repo.repo.url,
                            "commit": traced_repo.repo.commit,
                            "file_path": self._get_file_path(traced_repo, thm),
                            "full_name": thm.theorem.full_name,
                            "start": list(thm.start),
                            "end": list(thm.end),
                            "traced_tactics": tactics,
                        }
                    )
                oup_path = split_dir / f"{name}.json"
                json.dump(data, oup_path.open("wt"))
                logger.info(f"{print_prefix} {len(theorems)} theorems and {num_tactics} tactics saved to {oup_path}")

    def _get_file_path(self, traced_repo: TracedRepo, thm: TracedTheorem) -> str:
        if thm.repo == traced_repo.repo:
            # The theorem belongs to the traced repo itself.
            return str(thm.theorem.file_path)
        else:
            # The theorem belongs to one of the dependencies.
            for name, dep in traced_repo.dependencies.items():
                if dep == thm.repo:
                    return f"{LEAN4_PACKAGES_DIR}/{name}/{thm.theorem.file_path}"
            raise ValueError(f"Unable to find the dependency {thm.repo}")

    def export_premises(self, traced_repo: TracedRepo, dst_path: Path, repo: str) -> None:
        """Export all premise definitions in a traced repo to ``dst_path``."""
        repo_info = self.database["repos"][repo]
        print_prefix = f"[{repo_info['full_name']}]"
        oup_path = dst_path / "corpus.jsonl"
        num_premises = 0

        with oup_path.open("wt") as oup:
            G = traced_repo.traced_files_graph
            logger.info(f"Printing1 {traced_repo}")
            logger.info(f"Printing {G}")
            for tf_node in reversed(list(nx.topological_sort(G))):
                tf = G.nodes[tf_node]["traced_file"]
                imports = [str(_) for _ in G.successors(tf_node)]
                premises = tf.get_premise_definitions()
                num_premises += len(premises)
                oup.write(json.dumps({"path": str(tf.path), "imports": imports, "premises": premises}) + "\n")
        length = len(traced_repo.traced_files)
        logger.info(f"{print_prefix} {num_premises} theorems/definitions from {length} files saved to {oup_path}")

    def export_licenses(self, traced_repo: TracedRepo, dst_path: Path) -> None:
        """Export the licenses of a traced repo and all its dependencies to ``dst_path``."""
        license_dir = dst_path / "licenses"
        license_dir.mkdir()
        all_repos = [traced_repo.repo] + list(traced_repo.dependencies.values())

        for repo in all_repos:
            lic = repo.get_license()
            if lic is None:
                continue
            with (license_dir / repo.name).open("wt") as oup:
                oup.write(lic)

        with (license_dir / "README.md").open("wt") as oup:
            oup.write(
                """This directory contains licenses of Lean repos used to generate this dataset. \\
                    The dataset itself is released under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)."""
            )

    def export_metadata(self, traced_repo: TracedRepo, dst_path: Path, **kwargs) -> None:
        """Export the metadata of a traced repo to ``dst_path''."""
        metadata = dict(kwargs)
        metadata["creation_time"] = str(datetime.now())
        metadata["from_repo"] = {
            "url": traced_repo.repo.url,
            "commit": traced_repo.repo.commit,
        }
        metadata["leandojo_version"] = lean_dojo.__version__
        json.dump(metadata, (dst_path / "metadata.json").open("wt"))


class LeanUniverseData:
    def __init__(
        self,
        cache_dir: Path,
        log_file: Path,
        dataset_export_dir: Path,
        raw_dataset_dir: Path,
        repos_dir: Path,
        timestamp: str,
        ld_cache_dir: Optional[Path],
        repos_excluded: Optional[list[str]] = None,
        repos_included: Optional[list[str]] = None,
        licesnes_excluded: Optional[list[str]] = None,
        max_num_repos: int = -1,
        language: str = "lean",
        query: str = "lean",
        ld_max_num_procs: int = -1,
        large_dataset: int = -1,
        large_dataset_test_val_percent: int = -1,
        medium_dataset: int = -1,
        medium_dataset_test_val_percent: int = -1,
        small_dataset: int = -1,
        small_dataset_test_val_percent: int = -1,
        test_val_min_size: int = -1,
    ):
        self.cache_dir = cache_dir
        logger.info(f"Loading dataset from {cache_dir}")
        self.timestamp = timestamp
        self.github = get_github()
        self.dataset = Dataset(
            cache_dir=cache_dir,
            log_file=log_file,
            dataset_export_dir=dataset_export_dir,
            raw_dataset_dir=raw_dataset_dir,
            repos_dir=repos_dir,
            timestamp=timestamp,
            github=self.github,
            repos_excluded=repos_excluded,
            repos_included=repos_included,
            licesnes_excluded=licesnes_excluded,
            ld_cache_dir=ld_cache_dir,
            ld_max_num_procs=ld_max_num_procs,
            large_dataset=large_dataset,
            large_dataset_test_val_percent=large_dataset_test_val_percent,
            medium_dataset=medium_dataset,
            medium_dataset_test_val_percent=medium_dataset_test_val_percent,
            small_dataset=small_dataset,
            small_dataset_test_val_percent=small_dataset_test_val_percent,
            test_val_min_size=test_val_min_size,
        )

        self.max_num = max_num_repos
        self.query = query
        self.language = language

        self.repos_excluded = repos_excluded
        self.repos_included = repos_included
        self.licenses_excluded = licesnes_excluded

    def fetch_repos_and_update_database(self):
        self.__pull_repos()

    def __pull_repos(self) -> None:
        """
        Pulls repositories from the Lean Universe.
        """
        repos = get_lean_repos(
            GITHUB=self.github,
            repos_included=self.repos_included,
            max_num=self.max_num,
            query=self.query,
            language=self.language,
        )
        self.dataset.filter_new_repos(repos)
        self.dataset.clone_or_pull_repos()

    def build_lake(self) -> None:
        self.dataset.build_lake()

    def run_leandojo(self) -> None:
        self.dataset.configure_leandojo()
        self.dataset.run_leandojo()

    def report(self) -> None:
        self.dataset.report()
        logger.info("Report saved.")
