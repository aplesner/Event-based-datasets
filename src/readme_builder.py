"""
README generation for HuggingFace dataset repositories.
"""


def create_readme(
    repo_id: str,
    description: str,
    features: dict,
    metadata_keys: set,
    total_samples: int,
    has_split: bool
) -> str:
    """
    Create README content for HuggingFace dataset.

    Args:
        repo_id: HuggingFace repo ID (e.g., 'username/dataset-name')
        description: Dataset description
        features: Dictionary of feature definitions
        metadata_keys: Set of metadata field names
        total_samples: Total number of samples
        has_split: Whether dataset has train/test splits

    Returns:
        README content as string
    """
    dataset_name = repo_id.split('/')[-1]

    readme = f"""---
license: mit
task_categories:
- image-classification
tags:
- events
- neuromorphic
---

# {dataset_name}

{description}

## Dataset Structure

- **x**: Event X coordinates (int16)
- **y**: Event Y coordinates (int16)
- **timestamp**: Event timestamps in microseconds (uint32)
- **polarity**: Event polarity ON/OFF (bool)
- **label**: Sample label (string)
"""

    # Add metadata fields section
    if metadata_keys:
        readme += "\n### Metadata Fields\n\n"
        for key in sorted(metadata_keys):
            readme += f"- **{key}**: {features[key]}\n"

    # Add statistics
    readme += f"\n## Statistics\n\n- Total Samples: {total_samples:,}\n"

    # Add usage example if splits exist
    if has_split:
        readme += f"""
## Usage

```python
from datasets import load_dataset

# Load train split
ds_train = load_dataset('{repo_id}', split='train')

# Load test split
ds_test = load_dataset('{repo_id}', split='test')
```
"""

    # Add citation section
    readme += """
## Citation

Please also cite the original dataset creators when using this dataset in your research. See https://github.com/aplesner/Event-based-datasets for more information.
"""

    return readme


def upload_readme(repo_id: str, readme_content: str) -> None:
    """
    Upload README to HuggingFace repository.

    Args:
        repo_id: HuggingFace repo ID
        readme_content: README content as string
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
