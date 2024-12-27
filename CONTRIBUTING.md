# Contributing to LeanUniverse
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.


## Pre-commit hooks
In order to ensure your code lints, there are pre-commit hooks configured in the repository which you can install.
After installation, they will automatically run each time you commit.
An abbreviated guide is given below; for more information, refer to [the offical pre-commit documentation](https://pre-commit.com/).

### Installation
```
pip install pre-commit
pre-commit install
```

### Usage
Just commit your changes:
```
git commit -m "My informative commit message"
```

If there was a failure, you will get feedback
```
[INFO] Initializing environment for https://github.com/PyCQA/flake8.
[INFO] Installing environment for https://github.com/pre-commit/pre-commit-hooks.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/PyCQA/flake8.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
Trim Trailing Whitespace.................................................Failed
- hook id: trailing-whitespace
- exit code: 1
- files were modified by this hook
Fixing examples/nllb/modeling/wmt15_benchmark/eval_langs2.sh
Fix End of Files.........................................................Failed
- hook id: end-of-file-fixer
- exit code: 1
- files were modified by this hook
Fixing examples/few_shot/scripts/schedule_jobs_few_shot.py
flake8...................................................................Passed
```

Certain hooks modify your files to comply.
To include these modifications, you will need to add them (i.e. `git add ...`) and commit again.
