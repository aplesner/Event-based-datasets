"""
Simple function to upload event datasets to HuggingFace.

Usage:
    from push_to_hf import push_to_huggingface
    from your_module import load_dataset, Datasets
    
    samples = load_dataset(Datasets.ASL, cython=True)
    push_to_huggingface(samples, "your-username/asl-dvs")
"""

from datasets import Dataset, Features, Sequence, Value
import numpy as np
import tqdm


def push_to_huggingface(
    samples: list[dict],
    repo_id: str,
    description: str | None = None,
    private: bool = False
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
    print(f"Converting {len(samples)} samples...")
    
    # Prepare data
    dataset_dict = {
        'x': [],
        'y': [],
        'timestamp': [],
        'polarity': [],
        'label': [],
    }
    
    # Collect all metadata keys from all samples
    metadata_keys = set()
    for sample in samples:
        if 'metadata' in sample:
            metadata_keys.update(sample['metadata'].keys())
    
    # Initialize metadata columns
    for key in metadata_keys:
        dataset_dict[key] = []
    
    # Fill data
    for sample in tqdm.tqdm(samples, desc="Collecting samples in dict"):
        data = sample['data']
        dataset_dict['x'].append(data['x'].tolist())
        dataset_dict['y'].append(data['y'].tolist())
        dataset_dict['timestamp'].append(data['timestamp'].tolist())
        dataset_dict['polarity'].append(data['polarity'].tolist())
        dataset_dict['label'].append(sample['label'])
        
        # Add metadata
        metadata = sample.get('metadata', {})
        for key in metadata_keys:
            value = metadata.get(key)
            dataset_dict[key].append(value)
    
    # Define features
    features = {
        'x': Sequence(Value('int16')),
        'y': Sequence(Value('int16')),
        'timestamp': Sequence(Value('uint32')),
        'polarity': Sequence(Value('bool')),
        'label': Value('string'),
    }
    
    # Add metadata features (auto-detect type)
    for key in metadata_keys:
        sample_value = dataset_dict[key][0]
        if isinstance(sample_value, bool):
            features[key] = Value('bool')
        elif isinstance(sample_value, int):
            features[key] = Value('int64')
        elif isinstance(sample_value, float):
            features[key] = Value('float64')
        else:
            features[key] = Value('string')
    
    # Create dataset
    print("Creating HuggingFace dataset...")
    dataset = Dataset.from_dict(dataset_dict, features=Features(features))
    
    # Push to hub
    print(f"Pushing to {repo_id}...")
    dataset.push_to_hub(repo_id, private=private)
    
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
        
        readme += f"\n## Statistics\n\n- Samples: {len(dataset):,}\n"
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
    
    print(f"✓ Done! View at: https://huggingface.co/datasets/{repo_id}")


# Example usage
if __name__ == "__main__":
    # Example: Create dummy data
    dummy_samples = []
    for i in range(3):
        sample = {
            'data': {
                'x': np.random.randint(0, 128, 100, dtype=np.int16),
                'y': np.random.randint(0, 128, 100, dtype=np.int16),
                'timestamp': np.random.randint(0, 100000, 100, dtype=np.uint32),
                'polarity': np.random.randint(0, 2, 100, dtype=np.bool_)
            },
            'label': f'class_{i % 2}',
            'metadata': {
                'inverted': bool(i % 2),
                'split': 'train' if i < 2 else 'test'
            }
        }
        dummy_samples.append(sample)
    
    # Push to HuggingFace (uncomment to test)
    # push_to_huggingface(
    #     dummy_samples,
    #     "your-username/test-dataset",
    #     description="Test event dataset"
    # )
    
    print("To use with your data:")
    print("from push_to_hf import push_to_huggingface")
    print("from your_module import load_dataset, Datasets")
    print("samples = load_dataset(Datasets.ASL, cython=True)")
    print('push_to_huggingface(samples, "username/asl-dvs")')
