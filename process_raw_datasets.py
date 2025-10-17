from src.constants import Datasets
from src.data_loading import load_dataset
from src.push_to_hf import push_to_huggingface

datasets_to_process = [
    Datasets.ASL,
    Datasets.CIFAR10DVS,
    Datasets.DVSGesture,
    Datasets.DVSLip,
    Datasets.NCALTECH101,
    Datasets.NMNIST,
]

dataset_name = Datasets.NMNIST

dataset = load_dataset(dataset_name)

push_to_huggingface(
    samples=dataset,
    repo_id=f"aplesner-eth/{dataset_name}-dataset",
    description=f"{dataset_name} event-based dataset loaded and processed with custom data loading scripts. Not affiliated with the original creators. Created by A. Plesner to facilitate our group's research in event-based vision.",
    private=True,
)
