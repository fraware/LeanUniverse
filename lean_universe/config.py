# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Modern configuration management for LeanUniverse."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    url: str = Field(default="sqlite:///lean_universe.db", description="Database connection URL")
    echo: bool = Field(default=False, description="Enable SQLAlchemy echo mode")
    pool_size: int = Field(default=10, description="Database connection pool size")
    max_overflow: int = Field(default=20, description="Maximum database connection overflow")
    pool_pre_ping: bool = Field(default=True, description="Enable connection pool pre-ping")


class RedisConfig(BaseModel):
    """Redis configuration settings."""

    url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    max_connections: int = Field(default=20, description="Maximum Redis connections")
    socket_timeout: float = Field(default=5.0, description="Redis socket timeout")
    socket_connect_timeout: float = Field(default=2.0, description="Redis connection timeout")


class GitHubConfig(BaseModel):
    """GitHub API configuration."""

    access_token: Optional[str] = Field(default=None, description="GitHub personal access token")
    rate_limit_retry_attempts: int = Field(default=3, description="Number of retry attempts for rate limits")
    rate_limit_backoff_factor: float = Field(default=2.0, description="Backoff factor for rate limit retries")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent GitHub API requests")


class LeanDojoConfig(BaseModel):
    """LeanDojo configuration settings."""

    max_num_procs: int = Field(default=32, description="Maximum number of LeanDojo processes")
    cache_dir: Optional[Path] = Field(default=None, description="LeanDojo cache directory")
    timeout: int = Field(default=300, description="LeanDojo operation timeout in seconds")
    memory_limit: str = Field(default="8GB", description="Memory limit for LeanDojo processes")


class DatasetConfig(BaseModel):
    """Dataset generation configuration."""

    large_dataset_threshold: int = Field(default=100000, description="Threshold for large datasets")
    medium_dataset_threshold: int = Field(default=1000, description="Threshold for medium datasets")
    small_dataset_threshold: int = Field(default=100, description="Threshold for small datasets")

    large_dataset_test_val_percent: int = Field(default=2, description="Test/validation percentage for large datasets")
    medium_dataset_test_val_percent: int = Field(
        default=7, description="Test/validation percentage for medium datasets"
    )
    small_dataset_test_val_percent: int = Field(default=10, description="Test/validation percentage for small datasets")

    test_val_min_size: int = Field(default=5, description="Minimum size for test/validation sets")

    @field_validator(
        "large_dataset_test_val_percent", "medium_dataset_test_val_percent", "small_dataset_test_val_percent"
    )
    @classmethod
    def validate_percentages(cls, v: int) -> int:
        """Validate that percentages are between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        return v


class MLConfig(BaseModel):
    """Machine learning configuration."""

    model_cache_dir: Path = Field(default=Path("models"), description="Directory for caching ML models")
    batch_size: int = Field(default=32, description="Default batch size for ML operations")
    max_sequence_length: int = Field(default=2048, description="Maximum sequence length for models")
    device: str = Field(default="auto", description="Device for ML operations (auto, cpu, cuda)")
    precision: str = Field(default="float16", description="Precision for ML operations")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device specification."""
        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    enable_prometheus: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    tracing_endpoint: Optional[str] = Field(default=None, description="OpenTelemetry tracing endpoint")
    log_level: str = Field(default="INFO", description="Logging level")
    structured_logging: bool = Field(default=True, description="Enable structured logging")


class SecurityConfig(BaseModel):
    """Security configuration."""

    enable_encryption: bool = Field(default=False, description="Enable data encryption")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key for sensitive data")
    allowed_licenses: List[str] = Field(default_factory=list, description="List of allowed licenses")
    blocked_repositories: List[str] = Field(default_factory=list, description="List of blocked repositories")


class LeanUniverseConfig(BaseSettings):
    """Main configuration class for LeanUniverse."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", case_sensitive=False, extra="ignore"
    )

    # Core settings
    cache_dir: Path = Field(default=Path("cache"), description="Main cache directory")
    working_dir: Path = Field(default=Path.cwd(), description="Working directory")
    dataset_export_dir: Path = Field(default=Path("datasets"), description="Dataset export directory")
    raw_dataset_dir: Path = Field(default=Path("raw"), description="Raw dataset directory")
    repos_dir: Path = Field(default=Path("repos"), description="Repository directory")

    # Repository settings
    max_num_repos: int = Field(default=100, description="Maximum number of repositories to process")
    repos_included: List[str] = Field(default_factory=list, description="Repositories to include")
    repos_excluded: List[str] = Field(default_factory=list, description="Repositories to exclude")

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    lean_dojo: LeanDojoConfig = Field(default_factory=LeanDojoConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.cache_dir,
            self.working_dir,
            self.dataset_export_dir,
            self.raw_dataset_dir,
            self.repos_dir,
            self.ml.model_cache_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def log_file(self) -> Path:
        """Get the log file path."""
        return self.cache_dir / "logs" / "lean_universe.log"

    @property
    def database_file(self) -> Path:
        """Get the database file path."""
        return self.cache_dir / "database.json"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def validate_repository_url(self, url: str) -> bool:
        """Validate if a repository URL is allowed."""
        if url in self.security.blocked_repositories:
            return False
        return True

    def get_lean_dojo_cache_dir(self) -> Path:
        """Get the LeanDojo cache directory."""
        if self.lean_dojo.cache_dir:
            return self.lean_dojo.cache_dir
        return self.cache_dir / "lean_dojo"


# Global configuration instance
config = LeanUniverseConfig()


def get_config() -> LeanUniverseConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs: Any) -> None:
    """Update the global configuration."""
    global config
    config = LeanUniverseConfig(**{**config.model_dump(), **kwargs})


def load_config_from_file(config_path: Union[str, Path]) -> LeanUniverseConfig:
    """Load configuration from a file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # This would need to be implemented based on the file format (JSON, YAML, etc.)
    # For now, we'll use environment variables
    return LeanUniverseConfig(_env_file=str(config_path))
