"""
Simple function to upload event datasets to HuggingFace.

Usage:
    from push_to_hf import push_to_huggingface
    from constants import Datasets

    push_to_huggingface(Datasets.ASL, "your-username/asl-dvs")
"""

import os
from datasets import Dataset, Features, Sequence, Value, DatasetDict, Split
import numpy as np
import pandas as pd
import tqdm

from .constants import Datasets
from .load_data import load_aedat as cython_load_aedat
from .load_data import load_aedat4 as cython_load_aedat4
from .load_data import load_npy as cython_load_npy
from .load_data import load_binary as cython_load_binary


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

    # Determine metadata keys and types based on dataset
    metadata_keys, metadata_types = _get_metadata_schema(dataset)

    # Define features
    features = {
        'x': Sequence(Value('int16')),
        'y': Sequence(Value('int16')),
        'timestamp': Sequence(Value('uint32')),
        'polarity': Sequence(Value('bool')),
        'label': Value('string'),
    }
    features.update(metadata_types)

    # Check if dataset has train/test splits
    has_split = dataset in [Datasets.DVSGesture, Datasets.DVSLip, Datasets.NMNIST]

    if has_split:
        # Create DatasetDict with separate train/test splits
        print(f"Processing {dataset.value} with train/test splits...")

        train_gen = lambda: _file_based_generator(dataset, data_dir, split_filter='train')
        test_gen = lambda: _file_based_generator(dataset, data_dir, split_filter='test')

        train_dataset = Dataset.from_generator(train_gen, features=Features(features), split=Split.TRAIN)
        test_dataset = Dataset.from_generator(test_gen, features=Features(features), split=Split.TEST)

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        print(f"Pushing to {repo_id}...")
        dataset_dict.push_to_hub(repo_id, private=private)  # type: ignore

        total_samples = len(train_dataset) + len(test_dataset)  # type: ignore
    else:
        # Single dataset without splits (use 'train' as default split)
        print(f"Processing {dataset.value}...")

        gen = lambda: _file_based_generator(dataset, data_dir)
        hf_dataset = Dataset.from_generator(gen, features=Features(features), split=Split.TRAIN)

        print(f"Pushing to {repo_id}...")
        # Use smaller shard size for ASL to avoid OOM
        if dataset == Datasets.ASL:
            hf_dataset.push_to_hub(repo_id, private=private, max_shard_size="500MB")  # type: ignore
        else:
            hf_dataset.push_to_hub(repo_id, private=private)  # type: ignore

        total_samples = len(hf_dataset)  # type: ignore

    # Create README if description provided
    if description:
        _create_readme(repo_id, description, features, metadata_keys, total_samples, has_split)

    print(f"✓ Done! View at: https://huggingface.co/datasets/{repo_id}")


def _get_metadata_schema(dataset: Datasets) -> tuple[set[str], dict[str, Value]]:
    """Determine metadata keys and types for a dataset."""
    if dataset == Datasets.DVSGesture:
        return {'user', 'lighting'}, {
            'user': Value('string'),
            'lighting': Value('string'),
        }
    elif dataset == Datasets.DVSLip:
        return set(), {}
    elif dataset == Datasets.NMNIST:
        return set(), {}
    elif dataset == Datasets.POKERDVS:
        return {'inverted'}, {'inverted': Value('bool')}
    else:
        return set(), {}


def _file_based_generator(dataset: Datasets, data_dir: str, split_filter: str | None = None):
    """
    Generator that loads files on-the-fly and yields samples.
    Only loads one file into memory at a time.

    Args:
        dataset: Dataset enum
        data_dir: Root directory containing datasets
        split_filter: Optional filter for 'train' or 'test' split
    """
    if dataset == Datasets.ASL:
        for subject_id in range(1, 6):
            asl_path = os.path.join(data_dir, dataset.value, f"subject{subject_id}")
            files = sorted(os.listdir(asl_path))
            for file in tqdm.tqdm(files, desc=f"Loading ASL subject{subject_id}"):
                filepath = os.path.join(asl_path, file)
                data = cython_load_aedat(filepath)
                label = os.path.basename(file).split(".")[0]
                yield {
                    'x': data['x'].tolist(),
                    'y': data['y'].tolist(),
                    'timestamp': data['timestamp'].tolist(),
                    'polarity': data['polarity'].tolist(),
                    'label': label,
                }

    elif dataset == Datasets.CIFAR10DVS:
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        for class_name in classes:
            class_path = os.path.join(data_dir, dataset.value, class_name)
            files = sorted(os.listdir(class_path))
            for file in tqdm.tqdm(files, desc=f"Loading CIFAR10-DVS {class_name}"):
                filepath = os.path.join(class_path, file)
                data = cython_load_aedat4(filepath)
                yield {
                    'x': data['x'].tolist(),
                    'y': data['y'].tolist(),
                    'timestamp': data['timestamp'].tolist(),
                    'polarity': data['polarity'].tolist(),
                    'label': class_name,
                }

    elif dataset == Datasets.DVSGesture:
        # Load split information
        with open(os.path.join(data_dir, dataset.value, "trials_to_train.txt"), 'r') as f:
            train_split = [line.strip() for line in f.readlines()]
        with open(os.path.join(data_dir, dataset.value, "trials_to_test.txt"), 'r') as f:
            test_split = [line.strip() for line in f.readlines()]

        dvsgesture_path = os.path.join(data_dir, dataset.value, "DvsGesture")
        files = sorted(os.listdir(dvsgesture_path))

        for file in tqdm.tqdm(files, desc="Loading DVS-Gesture"):
            # Determine split
            if file in train_split:
                split = "train"
            elif file in test_split:
                split = "test"
            else:
                continue

            # Apply split filter
            if split_filter and split != split_filter:
                continue

            filepath = os.path.join(dvsgesture_path, file)
            data = cython_load_aedat(filepath)

            # Sort by timestamps
            timestamps = data["timestamp"]
            x = data['x']
            y = data['y']
            polarity = data['polarity']
            sorted_indices = np.argsort(timestamps)
            x = x[sorted_indices]
            y = y[sorted_indices]
            timestamps = timestamps[sorted_indices]
            polarity = polarity[sorted_indices]

            # Load labels
            label_filepath = filepath.replace("DvsGesture", "labels").replace(".aedat", "_labels.csv")
            df = pd.read_csv(label_filepath)
            labels = [(row['startTime_usec'], row['endTime_usec'], row['class']) for index, row in df.iterrows()]

            # Extract metadata
            user, lighting = os.path.basename(filepath).replace(".aedat", "").split("_", maxsplit=1)

            # Vectorized time indexing
            all_times = np.array([time for start_time, end_time, label in labels for time in (start_time, end_time)])
            all_indices = np.searchsorted(timestamps, all_times)
            time_to_index = {time: idx for time, idx in zip(all_times, all_indices)}

            # Yield multiple samples per file
            for start_time, end_time, label in labels:
                start_idx = time_to_index[start_time]
                end_idx = time_to_index[end_time]
                yield {
                    'x': x[start_idx:end_idx].tolist(),
                    'y': y[start_idx:end_idx].tolist(),
                    'timestamp': timestamps[start_idx:end_idx].tolist(),
                    'polarity': polarity[start_idx:end_idx].tolist(),
                    'label': label,
                    'user': user,
                    'lighting': lighting,
                }

    elif dataset == Datasets.DVSLip:
        for split in ["train", "test"]:
            # Apply split filter
            if split_filter and split != split_filter:
                continue

            for class_name in os.listdir(os.path.join(data_dir, dataset.value, split)):
                dvslip_path = os.path.join(data_dir, dataset.value, split, class_name)
                files = sorted(os.listdir(dvslip_path))
                for file in tqdm.tqdm(files, desc=f"Loading DVS-Lip {split}/{class_name}"):
                    filepath = os.path.join(dvslip_path, file)
                    data = cython_load_npy(filepath)
                    yield {
                        'x': data['x'].tolist(),
                        'y': data['y'].tolist(),
                        'timestamp': data['timestamp'].tolist(),
                        'polarity': data['polarity'].tolist(),
                        'label': class_name,
                    }

    elif dataset == Datasets.NCALTECH101:
        for class_name in os.listdir(os.path.join(data_dir, dataset.value)):
            class_path = os.path.join(data_dir, dataset.value, class_name)
            if not os.path.isdir(class_path):
                continue
            files = sorted(os.listdir(class_path))
            for file in tqdm.tqdm(files, desc=f"Loading N-Caltech101 {class_name}"):
                filepath = os.path.join(class_path, file)
                data = cython_load_binary(filepath)
                yield {
                    'x': data['x'].tolist(),
                    'y': data['y'].tolist(),
                    'timestamp': data['timestamp'].tolist(),
                    'polarity': data['polarity'].tolist(),
                    'label': class_name,
                }

    elif dataset == Datasets.NMNIST:
        for split in ["Train", "Test"]:
            # Apply split filter
            split_lower = split.lower()
            if split_filter and split_lower != split_filter:
                continue

            for class_id in range(10):
                class_path = os.path.join(data_dir, dataset.value, split, str(class_id))
                files = sorted(os.listdir(class_path))
                for file in tqdm.tqdm(files, desc=f"Loading N-MNIST {split}/{class_id}"):
                    filepath = os.path.join(class_path, file)
                    data = cython_load_binary(filepath)
                    yield {
                        'x': data['x'].tolist(),
                        'y': data['y'].tolist(),
                        'timestamp': data['timestamp'].tolist(),
                        'polarity': data['polarity'].tolist(),
                        'label': str(class_id),
                    }

    elif dataset == Datasets.POKERDVS:
        poker_dvs_path = os.path.join(data_dir, dataset.value)
        files = sorted(os.listdir(poker_dvs_path))
        for file in tqdm.tqdm(files, desc="Loading Poker-DVS"):
            filepath = os.path.join(poker_dvs_path, file)
            data = cython_load_aedat(filepath)
            label = os.path.basename(file).split(".")[0][1:]
            inverted = label.endswith('i')
            if inverted:
                label = label[:-1]
            # strip ints from end of label
            label = ''.join(filter(str.isalpha, label))
            yield {
                'x': data['x'].tolist(),
                'y': data['y'].tolist(),
                'timestamp': data['timestamp'].tolist(),
                'polarity': data['polarity'].tolist(),
                'label': label,
                'inverted': inverted,
            }


def _create_readme(repo_id: str, description: str, features: dict, metadata_keys: set, total_samples: int, has_split: bool):
    """Create and upload README to HuggingFace."""
    from huggingface_hub import HfApi

    readme = f"""---
license: mit
task_categories:
- image-classification
tags:
- events
- neuromorphic
---

# {repo_id.split('/')[-1]}

{description}

## Dataset Structure

- **x**: Event X coordinates (int16)
- **y**: Event Y coordinates (int16)
- **timestamp**: Event timestamps in microseconds (uint32)
- **polarity**: Event polarity ON/OFF (bool)
- **label**: Sample label (string)
"""
    if metadata_keys:
        readme += "\n### Metadata Fields\n\n"
        for key in sorted(metadata_keys):
            readme += f"- **{key}**: {features[key]}\n"

    readme += f"\n## Statistics\n\n- Total Samples: {total_samples:,}\n"

    if has_split:
        readme += "\n## Usage\n\n```python\nfrom datasets import load_dataset\n\n# Load train split\nds_train = load_dataset('" + repo_id + "', split='train')\n\n# Load test split\nds_test = load_dataset('" + repo_id + "', split='test')\n```\n"

    readme += "\n## Citation\n\nPlease also cite the original dataset creators when using this dataset in your research. See https://github.com/aplesner/Event-based-datasets for more information.\n"

    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
