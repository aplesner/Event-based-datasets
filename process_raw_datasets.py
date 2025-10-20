from src.constants import Datasets
from src.push_to_hf import push_to_huggingface

datasets_to_process = [
    # Datasets.POKERDVS,
    # Datasets.DVSGesture,
    # Datasets.DVSLip,
    # Datasets.NCALTECH101,
    # Datasets.NMNIST,
    Datasets.CIFAR10DVS,
    Datasets.ASL,
]

for dataset_name in datasets_to_process:
    print(f"Processing dataset: {dataset_name.value}")

    push_to_huggingface(
        dataset=dataset_name,
        repo_id=f"aplesner-eth/{dataset_name.value}",
        description=f"{dataset_name.value} event-based dataset loaded and processed with custom data loading scripts. Not affiliated with the original creators. Created by Andreas Plesner to facilitate our group's research in event-based vision. See https://github.com/aplesner/Event-based-datasets for more information and the source code used.",
        private=True,
    )
