"""
Shared utility functions for dataset processing.
"""

import os
import numpy as np


def convert_to_sample_dict(data: dict[str, np.ndarray], label: str, metadata: dict | None = None) -> dict:
    """
    Convert raw event data to HuggingFace-compatible sample dictionary.

    Args:
        data: Dictionary with keys 'x', 'y', 'timestamp', 'polarity' as numpy arrays
        label: Sample label (class name or identifier)
        metadata: Optional metadata dictionary

    Returns:
        Dictionary with all arrays converted to lists and metadata included
    """
    sample = {
        'x': data['x'].tolist(),
        'y': data['y'].tolist(),
        'timestamp': data['timestamp'].tolist(),
        'polarity': data['polarity'].tolist(),
        'label': label,
    }

    if metadata:
        sample.update(metadata)

    return sample


def load_split_files(dataset_path: str, train_file: str, test_file: str) -> tuple[list[str], list[str]]:
    """
    Load train/test split file lists.

    Args:
        dataset_path: Root path to dataset directory
        train_file: Filename containing train split (e.g., 'trials_to_train.txt')
        test_file: Filename containing test split (e.g., 'trials_to_test.txt')

    Returns:
        Tuple of (train_files, test_files) as lists of filenames
    """
    with open(os.path.join(dataset_path, train_file), 'r') as f:
        train_split = [line.strip() for line in f.readlines()]
    with open(os.path.join(dataset_path, test_file), 'r') as f:
        test_split = [line.strip() for line in f.readlines()]
    return train_split, test_split


def parse_label_from_filename(filename: str, strip_extension: bool = True) -> str:
    """
    Extract label from filename.

    Args:
        filename: Full filename (can include path)
        strip_extension: Whether to remove file extension

    Returns:
        Label extracted from filename
    """
    basename = os.path.basename(filename)
    if strip_extension:
        label = basename.split(".")[0]
    else:
        label = basename
    return label


def parse_poker_label(filename: str) -> tuple[str, bool]:
    """
    Parse PokerDVS label and inverted flag from filename.

    Args:
        filename: Poker-DVS filename (e.g., 'c2i_123.aedat')

    Returns:
        Tuple of (label, inverted_flag)
    """
    label = os.path.basename(filename).split(".")[0][1:]  # Remove leading char
    inverted = label.endswith('i')
    if inverted:
        label = label[:-1]
    # Strip trailing digits
    label = ''.join(filter(str.isalpha, label))
    return label, inverted


def get_files_in_directory(directory: str, sorted_files: bool = True) -> list[str]:
    """
    Get list of files in a directory.

    Args:
        directory: Directory path
        sorted_files: Whether to sort files alphabetically

    Returns:
        List of full file paths
    """
    files = os.listdir(directory)
    if sorted_files:
        files = sorted(files)
    return [os.path.join(directory, f) for f in files]


def enumerate_class_files(data_dir: str, dataset_value: str, classes: list[str]) -> list[tuple[str, str]]:
    """
    Enumerate all files organized by class subdirectories.

    Args:
        data_dir: Root data directory
        dataset_value: Dataset name/value
        classes: List of class names (subdirectory names)

    Returns:
        List of (filepath, class_label) tuples
    """
    file_list = []
    for class_name in classes:
        class_path = os.path.join(data_dir, dataset_value, class_name)
        files = sorted(os.listdir(class_path))
        for file in files:
            filepath = os.path.join(class_path, file)
            file_list.append((filepath, class_name))
    return file_list
