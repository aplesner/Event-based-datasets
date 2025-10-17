"""
Simple function to upload event datasets to HuggingFace.

Usage:
    from push_to_hf import push_to_huggingface
    from your_module import load_dataset, Datasets
    
    samples = load_dataset(Datasets.ASL, cython=True)
    push_to_huggingface(samples, "your-username/asl-dvs")
"""

from datasets import Dataset, Features, Sequence, Value, Split
import tqdm


def push_to_huggingface(
    samples: list[dict],
    repo_id: str,
    description: str | None = None,
    private: bool = False,
    split: str = 'all',
) -> None:
    """
    Push event dataset to HuggingFace Hub.

    Args:
        samples: List of samples from load_dataset()
                 Each sample: {'data': {...}, 'label': str, 'metadata': {...}}
        repo_id: HuggingFace repo (e.g., 'username/dataset-name')
        description: Optional dataset description
        private: Make repository private

    Example:
        samples = load_dataset(Datasets.ASL, cython=True)
        push_to_huggingface(samples, "myuser/asl-dvs")
    """
    assert split in ['all', 'train', 'test'], "split must be 'all', 'train', or 'test'"
    if split == 'train':
        dataset_split = Split.TRAIN
    elif split == 'test':
        dataset_split = Split.TEST
    else:
        dataset_split = Split.ALL

    print(f"Converting {len(samples)} samples...")

    # Collect all metadata keys from all samples
    metadata_keys = set()
    for sample in samples:
        if 'metadata' in sample:
            metadata_keys.update(sample['metadata'].keys())

    # Infer metadata types from first sample
    metadata_types = {}
    for sample in samples:
        if 'metadata' in sample:
            for key in metadata_keys:
                if key not in metadata_types and key in sample['metadata']:
                    value = sample['metadata'][key]
                    if isinstance(value, bool):
                        metadata_types[key] = Value('bool')
                    elif isinstance(value, int):
                        metadata_types[key] = Value('int64')
                    elif isinstance(value, float):
                        metadata_types[key] = Value('float64')
                    else:
                        metadata_types[key] = Value('string')
            if len(metadata_types) == len(metadata_keys):
                break

    # Define features
    features = {
        'x': Sequence(Value('int16')),
        'y': Sequence(Value('int16')),
        'timestamp': Sequence(Value('uint32')),
        'polarity': Sequence(Value('bool')),
        'label': Value('string'),
    }
    features.update(metadata_types)

    # Generator function to yield samples one at a time
    def sample_generator():
        for sample in tqdm.tqdm(samples, desc="Processing samples"):
            data = sample['data']
            row = {
                'x': data['x'].tolist(),
                'y': data['y'].tolist(),
                'timestamp': data['timestamp'].tolist(),
                'polarity': data['polarity'].tolist(),
                'label': sample['label'],
            }
            # Add metadata
            metadata = sample.get('metadata', {})
            for key in metadata_keys:
                row[key] = metadata.get(key)
            yield row

    # Create dataset from generator
    print("Creating HuggingFace dataset...")
    dataset = Dataset.from_generator(
        sample_generator, 
        features=Features(features),
        split=dataset_split
    )
    
    # Push to hub
    print(f"Pushing to {repo_id}...")
    dataset.push_to_hub(repo_id, private=private)  # type: ignore
    
    # Create README if description provided
    if description:
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

        readme += f"\n## Statistics\n\n- Samples: {len(dataset):,}\n"  # type: ignore

        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    print(f"✓ Done! View at: https://huggingface.co/datasets/{repo_id}")
