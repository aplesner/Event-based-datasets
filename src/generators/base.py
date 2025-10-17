"""
Base generator interface and types.
"""

from typing import Iterator, Protocol
import numpy as np


class SampleGenerator(Protocol):
    """Protocol for dataset sample generators."""

    def __call__(
        self,
        data_dir: str,
        dataset_value: str,
        split_filter: str | None = None,
        **kwargs
    ) -> Iterator[dict]:
        """
        Generate samples for a dataset.

        Args:
            data_dir: Root directory containing datasets
            dataset_value: Dataset enum value (e.g., 'ASL', 'NMNIST')
            split_filter: Optional 'train' or 'test' filter
            **kwargs: Additional dataset-specific parameters

        Yields:
            Dictionary with keys: x, y, timestamp, polarity, label, and optional metadata
        """
        ...
