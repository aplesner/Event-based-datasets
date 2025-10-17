import os
import time

import numpy as np
import tqdm
import pandas as pd

from .constants import Datasets
from .load_data import load_aedat as cython_load_aedat
from .load_data import load_aedat4 as cython_load_aedat4
from .load_data import load_npy as cython_load_npy
from .load_data import load_binary as cython_load_binary
from .load_data_py import load_aedat, load_aedat4, load_npy, load_binary

DATA_DIR = "datasets/"

def load_sample(dataset: Datasets, index: int, data_dir: str = DATA_DIR, cython: bool = True) -> dict[str, np.ndarray]:
    if not index in range(1, 101):
        index = 0
    if dataset == Datasets.ASL:
        asl_path = os.path.join(data_dir, Datasets.ASL.value, "subject1")
        files = sorted(os.listdir(asl_path))
        filepath = os.path.join(asl_path, files[index])
        if cython:
            return cython_load_aedat(filepath)
        else:
            return load_aedat(filepath)
    elif dataset == Datasets.CIFAR10DVS:
        cifar10dvs_path = os.path.join(data_dir, Datasets.CIFAR10DVS.value, "airplane")
        files = sorted(os.listdir(cifar10dvs_path))
        filepath = os.path.join(cifar10dvs_path, files[index])
        if cython:
            return cython_load_aedat4(filepath)
        else:
            return load_aedat4(filepath)
    elif dataset == Datasets.DVSGesture:
        dvsgesture_path = os.path.join(data_dir, Datasets.DVSGesture.value, "DvsGesture")
        files = sorted(os.listdir(dvsgesture_path))
        filepath = os.path.join(dvsgesture_path, files[index])
        if cython:
            return cython_load_aedat(filepath)
        else:
            return load_aedat(filepath)
    elif dataset == Datasets.DVSLip:
        dvslip_path = os.path.join(data_dir, Datasets.DVSLip.value, "train", "accused")
        files = sorted(os.listdir(dvslip_path))
        filepath = os.path.join(dvslip_path, files[index])
        if cython:
            return cython_load_npy(filepath)
        else:
            return load_npy(filepath)
    elif dataset == Datasets.NCALTECH101:
        ncaltech101_path = os.path.join(data_dir, Datasets.NCALTECH101.value, "airplanes")
        files = sorted(os.listdir(ncaltech101_path))
        filepath = os.path.join(ncaltech101_path, files[index])
        if cython:
            return cython_load_binary(filepath)
        else:
            return load_binary(filepath)
    elif dataset == Datasets.NMNIST:
        nmnist_path = os.path.join(data_dir, Datasets.NMNIST.value, "Train", "0")
        files = sorted(os.listdir(nmnist_path))
        filepath = os.path.join(nmnist_path, files[index])
        if cython:
            return cython_load_binary(filepath)
        else:
            return load_binary(filepath)
    elif dataset == Datasets.POKERDVS:
        poker_dvs_path = os.path.join(data_dir, Datasets.POKERDVS.value)
        files = sorted(os.listdir(poker_dvs_path))
        filepath = os.path.join(poker_dvs_path, files[index])
        if cython:
            return cython_load_aedat(filepath)
        else:
            return load_aedat(filepath)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def load_dataset(dataset: Datasets, data_dir: str = DATA_DIR, cython: bool = True) -> list[dict[str, np.ndarray]]:
    files = []
    samples = []

    if dataset == Datasets.ASL:
        for subject_id in range(1, 6):
            asl_path = os.path.join(data_dir, Datasets.ASL.value, f"subject{subject_id}")
            files.extend([os.path.join(asl_path, file) for file in sorted(os.listdir(asl_path))])
        for file in tqdm.tqdm(files, desc="Loading ASL dataset"):
            data = load_aedat(file) if not cython else cython_load_aedat(file)
            label = os.path.basename(file).split(".")[0]
            samples.append({'data': data, 'label': label})

    elif dataset == Datasets.CIFAR10DVS:
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        for class_name in classes:
            class_path = os.path.join(data_dir, Datasets.CIFAR10DVS.value, class_name)
            files.extend([(os.path.join(class_path, file), class_name) for file in sorted(os.listdir(class_path))])
        for file, label in tqdm.tqdm(files, desc="Loading CIFAR10-DVS dataset"):
            data = load_aedat(file) if not cython else cython_load_aedat(file)
            samples.append({'data': data, 'label': label})

    elif dataset == Datasets.DVSGesture:
        with open(os.path.join(data_dir, Datasets.DVSGesture.value, "trials_to_train.txt"), 'r') as f:
            train_split = [line.strip() for line in f.readlines()]
        with open(os.path.join(data_dir, Datasets.DVSGesture.value, "trials_to_test.txt"), 'r') as f:
            test_split = [line.strip() for line in f.readlines()]
        dvsgesture_path = os.path.join(data_dir, Datasets.DVSGesture.value, "DvsGesture")
        files.extend([os.path.join(dvsgesture_path, file) for file in sorted(os.listdir(dvsgesture_path))])
        for filepath in tqdm.tqdm(files, desc="Loading DVS-Gesture dataset"):
            data = load_aedat(filepath) if not cython else cython_load_aedat(filepath)
            timestamps = data["timestamp"]
            x = data['x']
            y = data['y']
            polarity = data['polarity']
            # sort the arrays by timestamps
            sorted_indices = np.argsort(timestamps)
            x = x[sorted_indices]
            y = y[sorted_indices]
            timestamps = timestamps[sorted_indices]
            polarity = polarity[sorted_indices]

            label_filepath = filepath.replace("DvsGesture", "labels").replace(".aedat", "_labels.csv")
            df = pd.read_csv(label_filepath)
            labels = [(row['startTime_usec'], row['endTime_usec'], row['class']) for index, row in df.iterrows()]
            
            # metadata from filename
            user, lightning = os.path.basename(filepath).replace(".aedat", "").split("_", maxsplit=1)
            split = None
            if os.path.basename(filepath) in train_split:
                split = "train"
            elif os.path.basename(filepath) in test_split:
                split = "test"
            assert split is not None, f"File {os.path.basename(filepath)} not found in train or test split lists."

            # search for all start and end times at once
            all_times = np.array([time for start_time, end_time, label in labels for time in (start_time, end_time)])
            all_indices = np.searchsorted(timestamps, all_times)
            time_to_index = {time: idx for time, idx in zip(all_times, all_indices)}

            for start_time, end_time, label in labels:
                start_idx = time_to_index[start_time]
                end_idx = time_to_index[end_time]
                sample_data = {
                    'x': x[start_idx:end_idx],
                    'y': y[start_idx:end_idx],
                    'timestamp': timestamps[start_idx:end_idx],
                    'polarity': polarity[start_idx:end_idx]
                }
                samples.append({'data': sample_data, 'label': label, 'metadata': {'user': user, 'lighting': lightning, 'split': split}})
    
    elif dataset == Datasets.DVSLip:
        for split in ["train", "test"]:
            for class_name in os.listdir(os.path.join(data_dir, Datasets.DVSLip.value, split)):
                dvslip_path = os.path.join(data_dir, Datasets.DVSLip.value, split, class_name)
                files.extend([(os.path.join(dvslip_path, file), split, class_name) for file in sorted(os.listdir(dvslip_path))])
        for file, split, label in tqdm.tqdm(files, desc="Loading DVS-Lip dataset"):
            data = load_npy(file) if not cython else cython_load_npy(file)
            samples.append({'data': data, 'label': label, 'metadata': {'split': split}})

    elif dataset == Datasets.NCALTECH101:
        for class_name in os.listdir(os.path.join(data_dir, Datasets.NCALTECH101.value)):
            class_path = os.path.join(data_dir, Datasets.NCALTECH101.value, class_name)
            if not os.path.isdir(class_path):
                continue
            files.extend([(os.path.join(class_path, file), class_name) for file in sorted(os.listdir(class_path))])
        for file, label in tqdm.tqdm(files, desc="Loading N-Caltech101 dataset"):
            data = load_binary(file) if not cython else cython_load_binary(file)
            samples.append({'data': data, 'label': label})

    elif dataset == Datasets.NMNIST:
        for split in ["Train", "Test"]:
            for class_id in range(10):
                class_path = os.path.join(data_dir, Datasets.NMNIST.value, split, str(class_id))
                files.extend([(os.path.join(class_path, file), str(class_id), split) for file in sorted(os.listdir(class_path))])
        for file, label, split in tqdm.tqdm(files, desc="Loading N-MNIST dataset"):
            data = load_binary(file) if not cython else cython_load_binary(file)
            samples.append({'data': data, 'label': label, 'metadata': {'split': split}})

    elif dataset == Datasets.POKERDVS:
        poker_dvs_path = os.path.join(data_dir, Datasets.POKERDVS.value)
        files.extend([os.path.join(poker_dvs_path, file) for file in sorted(os.listdir(poker_dvs_path))])
        for file in tqdm.tqdm(files, desc="Loading Poker-DVS dataset"):
            data = load_aedat(file) if not cython else cython_load_aedat(file)
            label = os.path.basename(file).split(".")[0][1:]
            inverted = label.endswith('i')
            if inverted:
                label = label[:-1]
            # strip ints from end of label
            label = ''.join(filter(str.isalpha, label))
            samples.append({'data': data, 'label': label, 'metadata': {'inverted': inverted}})

    return samples


def compare(dataset: Datasets = Datasets.DVSGesture, index: int = 0, data_dir: str = DATA_DIR):
    t = time.time()
    sample = load_sample(dataset, index, data_dir, cython=False)
    time_standard = time.time() - t
    t = time.time()
    sample_cython = load_sample(dataset, index, data_dir, cython=True)
    time_cython = time.time() - t
    speedup = time_standard / time_cython if time_cython > 0 else float('inf')
    print(f"Standard load time: {time_standard:.4f} seconds. Cython load time: {time_cython:.4f} seconds. Speedup: {speedup:.2f}x")
    if np.all([
        np.array_equal(sample[key], sample_cython[key]) for key in sample
    ]):
        print("All arrays are identical.\n\n")
    else:
        for key in sample:
            if not np.array_equal(sample[key], sample_cython[key]):
                print(f"Difference found in {key}")
                print("Standard:", sample[key][:10])
                print("Cython:", sample_cython[key][:10])
            else:
                print(f"{key} arrays are identical")
