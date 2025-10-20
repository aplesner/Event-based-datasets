"""
Generator for simple datasets: POKERDVS, NCALTECH101, DVSLip, NMNIST.

These datasets have straightforward structure:
- One file = one sample
- Files organized by class or split
"""

import os
from typing import Generator
import tqdm
from ..constants import Datasets
from ..dataset_configs import get_config
from ..utils import convert_to_sample_dict, parse_poker_label


def generate_simple_samples(
    data_dir: str,
    dataset_value: str,
    split_filter: str | None = None,
    dataset_enum: Datasets | None = None,
) -> Generator[dict, None, None]:
    """
    Generate samples for simple datasets.

    Args:
        data_dir: Root directory containing datasets
        dataset_value: Dataset name (e.g., 'POKERDVS', 'NMNIST')
        split_filter: Optional 'train' or 'test' filter
        dataset_enum: Dataset enum for config lookup

    Yields:
        Sample dictionaries
    """
    config = get_config(dataset_enum)
    loader = config['loader']

    if dataset_enum == Datasets.POKERDVS:
        yield from _generate_pokerdvs(data_dir, dataset_value, loader)

    elif dataset_enum == Datasets.NCALTECH101:
        yield from _generate_ncaltech101(data_dir, dataset_value, loader)

    elif dataset_enum == Datasets.DVSLip:
        yield from _generate_dvslip(data_dir, dataset_value, loader, split_filter)

    elif dataset_enum == Datasets.NMNIST:
        yield from _generate_nmnist(data_dir, dataset_value, loader, split_filter, config)


def _generate_pokerdvs(data_dir: str, dataset_value: str, loader) -> Generator[dict, None, None]:
    """Generate PokerDVS samples."""
    poker_dvs_path = os.path.join(data_dir, dataset_value)
    files = sorted(os.listdir(poker_dvs_path))

    for file in tqdm.tqdm(files, desc="Loading Poker-DVS"):
        filepath = os.path.join(poker_dvs_path, file)
        data = loader(filepath)
        label, inverted = parse_poker_label(file)
        yield convert_to_sample_dict(data, label, {'inverted': inverted})


def _generate_ncaltech101(data_dir: str, dataset_value: str, loader) -> Generator[dict, None, None]:
    """Generate N-Caltech101 samples."""
    base_path = os.path.join(data_dir, dataset_value)

    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue

        files = sorted(os.listdir(class_path))
        for file in tqdm.tqdm(files, desc=f"Loading N-Caltech101 {class_name}"):
            filepath = os.path.join(class_path, file)
            data = loader(filepath)
            yield convert_to_sample_dict(data, class_name)


def _generate_dvslip(data_dir: str, dataset_value: str, loader, split_filter: str | None) -> Generator[dict, None, None]:
    """Generate DVS-Lip samples."""
    for split in ["train", "test"]:
        if split_filter and split != split_filter:
            continue

        split_path = os.path.join(data_dir, dataset_value, split)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            files = sorted(os.listdir(class_path))

            for file in tqdm.tqdm(files, desc=f"Loading DVS-Lip {split}/{class_name}"):
                filepath = os.path.join(class_path, file)
                data = loader(filepath)
                yield convert_to_sample_dict(data, class_name)


def _generate_nmnist(data_dir: str, dataset_value: str, loader, split_filter: str | None, config: dict) -> Generator[dict, None, None]:
    """Generate N-MNIST samples."""
    split_dirs = config.get('split_dirs', {'train': 'Train', 'test': 'Test'})
    num_classes = config.get('num_classes', 10)

    for split_key, split_dir in split_dirs.items():
        split_lower = split_key.lower()
        if split_filter and split_lower != split_filter:
            continue

        for class_id in range(num_classes):
            class_path = os.path.join(data_dir, dataset_value, split_dir, str(class_id))
            files = sorted(os.listdir(class_path))

            for file in tqdm.tqdm(files, desc=f"Loading N-MNIST {split_dir}/{class_id}"):
                filepath = os.path.join(class_path, file)
                data = loader(filepath)
                yield convert_to_sample_dict(data, str(class_id))
