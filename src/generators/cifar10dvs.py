"""
Generator for CIFAR10-DVS dataset.
"""

import os
import tqdm
from ..dataset_configs import get_config, CIFAR10_CLASSES
from ..constants import Datasets
from ..utils import convert_to_sample_dict


def generate_cifar10dvs_samples(
    data_dir: str,
    dataset_value: str,
    split_filter: str | None = None,
) -> dict:
    """
    Generate CIFAR10-DVS samples.

    Args:
        data_dir: Root directory containing datasets
        dataset_value: Dataset name ('CIFAR10DVS')
        split_filter: Not used (dataset has no splits)

    Yields:
        Sample dictionaries
    """
    config = get_config(Datasets.CIFAR10DVS)
    loader = config['loader']
    classes = config.get('classes', CIFAR10_CLASSES)

    for class_name in classes:
        class_path = os.path.join(data_dir, dataset_value, class_name)
        files = sorted(os.listdir(class_path))

        for file in tqdm.tqdm(files, desc=f"Loading CIFAR10-DVS {class_name}"):
            filepath = os.path.join(class_path, file)
            data = loader(filepath)
            yield convert_to_sample_dict(data, class_name)
