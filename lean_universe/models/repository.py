# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Repository models for LeanUniverse."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class RepositoryStatus(str, Enum):
    """Repository processing status."""

    DISCOVERED = "discovered"
    CLONED = "cloned"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RepositoryInfo(BaseModel):
    """Repository information model."""

    url: str = Field(description="Repository URL")
    name: str = Field(description="Repository name")
    full_name: str = Field(description="Full repository name (owner/repo)")
    description: str = Field(default="", description="Repository description")
    language: str = Field(default="Unknown", description="Primary language")
    stars: int = Field(default=0, description="Number of stars")
    forks: int = Field(default=0, description="Number of forks")
    created_at: datetime = Field(description="Creation date")
    updated_at: datetime = Field(description="Last update date")
    license: Optional[str] = Field(default=None, description="License SPDX ID")
    default_branch: str = Field(default="main", description="Default branch")
    is_fork: bool = Field(default=False, description="Whether repository is a fork")
    size: int = Field(default=0, description="Repository size in KB")
    status: RepositoryStatus = Field(default=RepositoryStatus.DISCOVERED, description="Processing status")

    # Local processing fields
    local_path: Optional[Path] = Field(default=None, description="Local repository path")
    latest_commit: Optional[str] = Field(default=None, description="Latest commit hash")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    # Lean4 specific fields
    lean_version: Optional[str] = Field(default=None, description="Lean version from toolchain")
    has_lakefile: bool = Field(default=False, description="Has lakefile.lean")
    has_toolchain: bool = Field(default=False, description="Has lean-toolchain")
    is_lean3: bool = Field(default=False, description="Is Lean3 repository")

    # Processing metadata
    discovered_at: datetime = Field(default_factory=datetime.now, description="Discovery timestamp")
    processed_at: Optional[datetime] = Field(default=None, description="Processing timestamp")
    processing_duration: Optional[float] = Field(default=None, description="Processing duration in seconds")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        arbitrary_types_allowed = True
