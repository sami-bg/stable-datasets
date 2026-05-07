"""Utility functions for benchmarks."""

import os
from pathlib import Path


def get_data_dir(dataset_name: str = None) -> Path:
    """Get the data directory for storing datasets.

    The directory is determined in the following order:
    1. Environment variable ANONYMOUS_PRETRAINING_LIB_DATA_DIR if set
    2. Default to ~/.cache/anonymous-pretraining-lib/data

    Args:
        dataset_name: Optional name of the dataset to create a subdirectory

    Returns:
        Path object pointing to the data directory

    Examples:
        >>> # Get general data directory
        >>> data_dir = get_data_dir()

        >>> # Get CIFAR-10 specific directory
        >>> cifar10_dir = get_data_dir("cifar10")

        >>> # Set custom directory via environment variable
        >>> # export ANONYMOUS_PRETRAINING_LIB_DATA_DIR=/path/to/my/data
    """
    # Check for environment variable
    if "ANONYMOUS_PRETRAINING_LIB_DATA_DIR" in os.environ:
        base_dir = Path(os.environ["ANONYMOUS_PRETRAINING_LIB_DATA_DIR"])
    else:
        # Use default location in user's cache directory
        base_dir = Path.home() / ".cache" / "anonymous-pretraining-lib" / "data"

    # Create base directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)

    # Add dataset subdirectory if specified
    if dataset_name:
        data_dir = base_dir / dataset_name
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        data_dir = base_dir

    return data_dir
