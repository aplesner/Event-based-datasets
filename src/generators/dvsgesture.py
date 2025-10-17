"""
Generator for DVS-Gesture dataset.

Special handling: Each file contains multiple temporal segments.
Each segment is extracted based on CSV labels and yielded as a separate sample.
"""

import os
import numpy as np
import pandas as pd
import tqdm
from ..dataset_configs import get_config
from ..constants import Datasets
from ..utils import convert_to_sample_dict, load_split_files


def generate_dvsgesture_samples(
    data_dir: str,
    dataset_value: str,
    split_filter: str | None = None,
) -> dict:
    """
    Generate DVS-Gesture samples.

    Each file contains multiple gesture samples separated by timestamps.
    Labels are loaded from CSV files.

    Args:
        data_dir: Root directory containing datasets
        dataset_value: Dataset name ('DVSGesture')
        split_filter: Optional 'train' or 'test' filter

    Yields:
        Sample dictionaries with temporal segments
    """
    config = get_config(Datasets.DVSGesture)
    loader = config['loader']
    split_files = config['split_files']

    # Load train/test split file lists
    dataset_path = os.path.join(data_dir, dataset_value)
    train_split, test_split = load_split_files(
        dataset_path,
        split_files['train'],
        split_files['test']
    )

    # Process files
    dvsgesture_path = os.path.join(dataset_path, "DvsGesture")
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

        # Load and process file
        filepath = os.path.join(dvsgesture_path, file)
        yield from _process_dvsgesture_file(filepath, loader, split)


def _process_dvsgesture_file(filepath: str, loader, split: str) -> dict:
    """
    Process a single DVS-Gesture file and yield multiple samples.

    Args:
        filepath: Path to .aedat file
        loader: Data loader function
        split: 'train' or 'test'

    Yields:
        Sample dictionaries for each temporal segment
    """
    # Load raw data
    data = loader(filepath)

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

    # Load temporal segment labels
    label_filepath = filepath.replace("DvsGesture", "labels").replace(".aedat", "_labels.csv")
    df = pd.read_csv(label_filepath)
    labels = [(row['startTime_usec'], row['endTime_usec'], row['class']) for _, row in df.iterrows()]

    # Extract metadata from filename
    filename = os.path.basename(filepath).replace(".aedat", "")
    user, lighting = filename.split("_", maxsplit=1)

    # Vectorized time indexing for efficiency
    all_times = np.array([
        time
        for start_time, end_time, _ in labels
        for time in (start_time, end_time)
    ])
    all_indices = np.searchsorted(timestamps, all_times)
    time_to_index = {time: idx for time, idx in zip(all_times, all_indices)}

    # Yield each temporal segment as a sample
    for start_time, end_time, label in labels:
        start_idx = time_to_index[start_time]
        end_idx = time_to_index[end_time]

        segment_data = {
            'x': x[start_idx:end_idx],
            'y': y[start_idx:end_idx],
            'timestamp': timestamps[start_idx:end_idx],
            'polarity': polarity[start_idx:end_idx]
        }

        metadata = {
            'user': user,
            'lighting': lighting,
        }

        yield convert_to_sample_dict(segment_data, label, metadata)
