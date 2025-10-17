from src.constants import Datasets
from src.data_loading import load_dataset
from src.push_to_hf import push_to_huggingface

datasets_to_process = [
    # Datasets.ASL,
    # Datasets.CIFAR10DVS,
    # Datasets.DVSGesture,
    # Datasets.DVSLip,
    Datasets.NCALTECH101,
    # Datasets.NMNIST,
    Datasets.POKERDVS,
]

for dataset_name in datasets_to_process:
    print(f"Processing dataset: {dataset_name.value}")

    dataset = load_dataset(dataset_name)

    print(f"Pushing dataset {dataset_name.value} with {len(dataset)} samples to HuggingFace Hub...")

    push_to_huggingface(
        samples=dataset,
        repo_id=f"aplesner-eth/{dataset_name.value}",
        description=f"{dataset_name.value} event-based dataset loaded and processed with custom data loading scripts. Not affiliated with the original creators. Created by Andreas Plesner to facilitate our group's research in event-based vision. See https://github.com/aplesner/Event-based-datasets for more information and the source code used.",
        private=True,
        split='all',
    )
