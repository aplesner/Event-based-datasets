"""
Generator for ASL-DVS dataset.

Special handling: Processes by subject to avoid out-of-memory issues.
"""

import os
import tqdm
from ..dataset_configs import get_config
from ..constants import Datasets
from ..utils import convert_to_sample_dict, parse_label_from_filename


def generate_asl_samples(
    data_dir: str,
    dataset_value: str,
    split_filter: str | None = None,
    subject_filter: int | None = None,
) -> dict:
    """
    Generate ASL-DVS samples.

    Args:
        data_dir: Root directory containing datasets
        dataset_value: Dataset name ('ASL')
        split_filter: Not used (dataset has no splits)
        subject_filter: Optional subject ID (1-5) to process only one subject

    Yields:
        Sample dictionaries
    """
    config = get_config(Datasets.ASL)
    loader = config['loader']
    num_subjects = config.get('num_subjects', 5)

    # Process specific subject or all subjects
    subjects = [subject_filter] if subject_filter else range(1, num_subjects + 1)

    for subject_id in subjects:
        subject_path = os.path.join(data_dir, dataset_value, f"subject{subject_id}")
        files = sorted(os.listdir(subject_path))

        for file in tqdm.tqdm(files, desc=f"Loading ASL subject{subject_id}"):
            filepath = os.path.join(subject_path, file)
            data = loader(filepath)
            label = parse_label_from_filename(file)
            yield convert_to_sample_dict(data, label)
