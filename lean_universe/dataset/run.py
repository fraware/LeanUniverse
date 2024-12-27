# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from pathlib import Path

from lean_universe.dataset.args import EvalArgs
from lean_universe.dataset.data import LeanUniverseData
from lean_universe.utils.logger import add_logger_file_handler, init_logger, log_host
from lean_universe.utils.params import cfg_from_cli


def main(args: EvalArgs):
    init_logger()
    log_host()
    add_logger_file_handler(args.log_file)
    logger = getLogger()
    logger.info(f"Logger file is located at {args.log_file}")


if __name__ == "__main__":
    args: EvalArgs = cfg_from_cli(schema=EvalArgs)
    main(args)
    logger = getLogger()

    data = LeanUniverseData(
        cache_dir=Path(args.cache_dir),
        log_file=Path(args.log_file),
        dataset_export_dir=Path(args.dataset_export_dir),
        raw_dataset_dir=Path(args.raw_dataset_dir),
        repos_dir=Path(args.repos_dir),
        timestamp=args.timestamp,
        repos_excluded=[
            "https://github.com/leanprover-community/mathlib4_with_LeanInfer",
        ],
        repos_included=args.repos_included,
        # repos_included=[
        #     # "https://github.com/dwrensha/compfiles",
        #     # "https://github.com/goens/lost-pop-lean",
        #     # "https://github.com/RustyYato/lean-algebra",
        #     # "https://github.com/Junology/dijkstra",
        #     # "https://github.com/arthurpaulino/viper",
        #     # "https://github.com/isubasinghe/leftpad-lean",
        #     # "https://github.com/FizzyElt/lean-pratice",
        # ],
        max_num_repos=args.max_num_repos,
        # max_num_repos=1,
        ld_max_num_procs=args.ld_max_num_procs,
        # ld_max_num_procs=100,
        ld_cache_dir=Path(args.cache_dir) / "ld_cache",
        large_dataset=args.large_dataset,
        large_dataset_test_val_percent=args.large_dataset_test_val_percent,
        medium_dataset=args.medium_dataset,
        medium_dataset_test_val_percent=args.medium_dataset_test_val_percent,
        small_dataset=args.small_dataset,
        small_dataset_test_val_percent=args.small_dataset_test_val_percent,
        test_val_min_size=args.test_val_min_size,
    )

    data.fetch_repos_and_update_database()
    # data.build_lake()
    data.run_leandojo()

    data.report()
