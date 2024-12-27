# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import socket
import sys
import time
from datetime import timedelta
from functools import cache


@cache
def get_is_dora_run() -> bool:
    # Set in apps.codegen_dora.compat.py
    return os.environ.get("DORA_FORCE_DISTRIB") is not None


@cache
def get_is_local_run() -> bool:
    return os.environ.get("LOCAL_RUN") is not None


@cache
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_local_run()


@cache
def get_global_rank() -> int:
    if get_is_dora_run():
        return int(os.environ["RANK"])
    elif get_is_local_run():
        return 0
    elif get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


def log_host():
    logger = logging.getLogger()
    logger.info(f"Host: {socket.gethostname()}")
    logger.info(f"Job hosts: {os.environ.get('SLURM_JOB_NODELIST', '')}")
    logger.info(f"Slurm job id: {int(os.environ.get('SLURM_JOB_ID', -1))}")


class LogFormatter(logging.Formatter):
    def __init__(self):
        self.start_time = time.time()
        self.rank = get_global_rank()
        self.show_rank = not get_is_slurm_job()  # srun has --label

    def format(self, record):
        # define prefix
        # record.pathname / record.filename / record.lineno
        subsecond, seconds = math.modf(record.created)
        curr_date = time.strftime("%y-%m-%d %H:%M:%S", time.localtime(seconds)) + f".{int(subsecond * 1_000_000):06d}"
        delta = timedelta(seconds=round(record.created - self.start_time))
        if self.show_rank:
            prefix = f"{self.rank}: {record.levelname:<7} {curr_date} - {delta} - "
        else:
            prefix = f"{record.levelname:<7} {curr_date} - {delta} - "

        # logged content
        content = record.getMessage()
        indent = " " * len(prefix)
        content = content.replace("\n", "\n" + indent)

        # Exception handling as in the default formatter, albeit with indenting
        # according to our custom prefix
        if record.exc_info and not record.exc_text:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            content = content + indent.join([line + "\n" for line in record.exc_text.splitlines()])
            if content[-1:] == "\n":
                content = content[:-1]
        if record.stack_info:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            stack_text = self.formatStack(record.stack_info)
            content = content + indent.join([line + "\n" for line in stack_text.splitlines()])
            if content[-1:] == "\n":
                content = content[:-1]

        return prefix + content


def init_logger() -> logging.Logger:
    # log everything
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.setFormatter(LogFormatter())

    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(LogFormatter())

    # set stream handlers
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)

    # turn package loggers silent
    logging.getLogger("filelock").setLevel(logging.WARNING)

    return logger


def add_logger_file_handler(filepath: str):
    # build file handler
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(LogFormatter())

    # update logger
    logger = logging.getLogger()
    logger.addHandler(file_handler)
