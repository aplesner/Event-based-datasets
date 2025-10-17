"""
Centralized dataset configurations.

Each dataset has a configuration dictionary with:
- loader: Function to load a single file
- has_split: Whether dataset has train/test splits
- metadata_fields: Dict of metadata field names to HuggingFace Value types
- Additional dataset-specific configuration
"""

from datasets import Value
from .constants import Datasets
from .load_data import (
    load_aedat as cython_load_aedat,
    load_aedat4 as cython_load_aedat4,
    load_npy as cython_load_npy,
    load_binary as cython_load_binary,
)


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


DATASET_CONFIGS = {
    Datasets.ASL: {
        'loader': cython_load_aedat,
        'has_split': False,
        'metadata_fields': {},
        'needs_chunking': True,  # Process by subject to avoid OOM
        'num_subjects': 5,
    },
    Datasets.CIFAR10DVS: {
        'loader': cython_load_aedat4,
        'has_split': False,
        'metadata_fields': {},
        'classes': CIFAR10_CLASSES,
    },
    Datasets.DVSGesture: {
        'loader': cython_load_aedat,
        'has_split': True,
        'metadata_fields': {
            'user': Value('string'),
            'lighting': Value('string'),
        },
        'split_files': {
            'train': 'trials_to_train.txt',
            'test': 'trials_to_test.txt',
        },
        'multi_sample_per_file': True,  # Each file contains multiple temporal segments
    },
    Datasets.DVSLip: {
        'loader': cython_load_npy,
        'has_split': True,
        'metadata_fields': {},
    },
    Datasets.NCALTECH101: {
        'loader': cython_load_binary,
        'has_split': False,
        'metadata_fields': {},
    },
    Datasets.NMNIST: {
        'loader': cython_load_binary,
        'has_split': True,
        'metadata_fields': {},
        'split_dirs': {'train': 'Train', 'test': 'Test'},
        'num_classes': 10,
    },
    Datasets.POKERDVS: {
        'loader': cython_load_aedat,
        'has_split': False,
        'metadata_fields': {
            'inverted': Value('bool'),
        },
    },
}


def get_config(dataset: Datasets) -> dict:
    """Get configuration for a dataset."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"No configuration found for dataset: {dataset}")
    return DATASET_CONFIGS[dataset]


def get_metadata_schema(dataset: Datasets) -> tuple[set[str], dict[str, Value]]:
    """Get metadata keys and types for a dataset."""
    config = get_config(dataset)
    fields = config.get('metadata_fields', {})
    return set(fields.keys()), fields
