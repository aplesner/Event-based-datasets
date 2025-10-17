"""
Upload event datasets to HuggingFace Hub using memory-efficient file-based loading.

Usage:
    from push_to_hf import push_to_huggingface
    from constants import Datasets

    push_to_huggingface(Datasets.ASL, "your-username/asl-dvs")
"""

import os
from datasets import Dataset, Features, Sequence, Value, DatasetDict, Split, concatenate_datasets

from .constants import Datasets
from .dataset_configs import get_config, get_metadata_schema
from .generators import (
    generate_asl_samples,
    generate_cifar10dvs_samples,
    generate_dvsgesture_samples,
    generate_simple_samples,
)
from .readme_builder import create_readme, upload_readme


DATA_DIR = "datasets/"


def push_to_huggingface(
    dataset: Datasets,
    repo_id: str,
    description: str | None = None,
    private: bool = False,
    data_dir: str = DATA_DIR,
) -> None:
    """
    Push event dataset to HuggingFace Hub using memory-efficient file-based loading.

    Args:
        dataset: Dataset enum (e.g., Datasets.ASL)
        repo_id: HuggingFace repo (e.g., 'username/dataset-name')
        description: Optional dataset description
        private: Make repository private
        data_dir: Root directory containing datasets

    Example:
        push_to_huggingface(Datasets.ASL, "myuser/asl-dvs")
    """
    config = get_config(dataset)
    metadata_keys, metadata_types = get_metadata_schema(dataset)

    # Define HuggingFace features schema
    features = _create_feature_schema(metadata_types)

    # Process dataset based on configuration
    if config.get('has_split'):
        hf_dataset, total_samples = _process_split_dataset(dataset, data_dir, features, config)
    elif config.get('needs_chunking'):
        hf_dataset, total_samples = _process_chunked_dataset(dataset, data_dir, features, config)
    else:
        hf_dataset, total_samples = _process_standard_dataset(dataset, data_dir, features)

    # Push to HuggingFace
    print(f"Pushing to {repo_id}...")
    if isinstance(hf_dataset, DatasetDict):
        hf_dataset.push_to_hub(repo_id, private=private)  # type: ignore
    else:
        hf_dataset.push_to_hub(repo_id, private=private)  # type: ignore

    # Create and upload README
    if description:
        readme_content = create_readme(
            repo_id, description, features, metadata_keys, total_samples, config.get('has_split', False)
        )
        upload_readme(repo_id, readme_content)

    print(f"✓ Done! View at: https://huggingface.co/datasets/{repo_id}")


def _create_feature_schema(metadata_types: dict) -> dict:
    """Create HuggingFace feature schema."""
    features = {
        'x': Sequence(Value('int16')),
        'y': Sequence(Value('int16')),
        'timestamp': Sequence(Value('uint32')),
        'polarity': Sequence(Value('bool')),
        'label': Value('string'),
    }
    features.update(metadata_types)
    return features


def _process_split_dataset(dataset: Datasets, data_dir: str, features: dict, config: dict) -> tuple:
    """Process dataset with train/test splits."""
    print(f"Processing {dataset.value} with train/test splits...")

    # Get generator function
    generator_fn = _get_generator_function(dataset)

    # Create train and test generators
    train_gen = lambda: generator_fn(data_dir, dataset.value, split_filter='train')
    test_gen = lambda: generator_fn(data_dir, dataset.value, split_filter='test')

    # Generate datasets
    train_dataset = Dataset.from_generator(train_gen, features=Features(features))
    test_dataset = Dataset.from_generator(test_gen, features=Features(features))

    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    total_samples = len(train_dataset) + len(test_dataset)  # type: ignore
    return dataset_dict, total_samples


def _process_chunked_dataset(dataset: Datasets, data_dir: str, features: dict, config: dict) -> tuple:
    """Process dataset in chunks (e.g., ASL by subject) to avoid OOM."""
    print(f"Processing {dataset.value} in chunks to avoid OOM...")

    generator_fn = _get_generator_function(dataset)
    num_subjects = config.get('num_subjects', 5)

    # Process each chunk
    chunk_datasets = []
    for subject_id in range(1, num_subjects + 1):
        print(f"Processing subject {subject_id}/{num_subjects}...")
        gen = lambda sid=subject_id: generator_fn(data_dir, dataset.value, subject_filter=sid)
        ds = Dataset.from_generator(gen, features=Features(features))
        chunk_datasets.append(ds)

    # Concatenate all chunks
    hf_dataset = concatenate_datasets(chunk_datasets)
    total_samples = len(hf_dataset)  # type: ignore
    return hf_dataset, total_samples


def _process_standard_dataset(dataset: Datasets, data_dir: str, features: dict) -> tuple:
    """Process standard dataset without splits or chunking."""
    print(f"Processing {dataset.value}...")

    generator_fn = _get_generator_function(dataset)
    gen = lambda: generator_fn(data_dir, dataset.value)

    hf_dataset = Dataset.from_generator(gen, features=Features(features), split=Split.TRAIN)
    total_samples = len(hf_dataset)  # type: ignore
    return hf_dataset, total_samples


def _get_generator_function(dataset: Datasets):
    """Get the appropriate generator function for a dataset."""
    if dataset == Datasets.ASL:
        return generate_asl_samples
    elif dataset == Datasets.CIFAR10DVS:
        return generate_cifar10dvs_samples
    elif dataset == Datasets.DVSGesture:
        return generate_dvsgesture_samples
    elif dataset in [Datasets.POKERDVS, Datasets.NCALTECH101, Datasets.DVSLip, Datasets.NMNIST]:
        # Simple generator handles multiple datasets
        return lambda *args, **kwargs: generate_simple_samples(*args, dataset_enum=dataset, **kwargs)
    else:
        raise ValueError(f"No generator found for dataset: {dataset}")
