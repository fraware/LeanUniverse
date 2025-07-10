# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build for LeanUniverse
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main --no-root

# Development stage
FROM base as development

# Install development dependencies
RUN poetry install --with dev

# Copy source code
COPY . .

# Install pre-commit hooks
RUN poetry run pre-commit install

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["poetry", "run", "lean-universe", "--help"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r leanuniverse && useradd -r -g leanuniverse leanuniverse

# Copy source code
COPY . .

# Install the package
RUN poetry install --only=main

# Change ownership
RUN chown -R leanuniverse:leanuniverse /app

# Switch to non-root user
USER leanuniverse

# Create necessary directories
RUN mkdir -p /app/cache /app/datasets /app/repos

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD poetry run python -c "import lean_universe; print('OK')" || exit 1

# Default command
CMD ["poetry", "run", "lean-universe", "init"]

# Testing stage
FROM development as testing

# Install test dependencies
RUN poetry install --with test

# Run tests
CMD ["poetry", "run", "pytest", "--cov=lean_universe", "--cov-report=html"]

# Documentation stage
FROM development as docs

# Install documentation dependencies
RUN poetry install --with docs

# Build documentation
CMD ["poetry", "run", "mkdocs", "build"] 