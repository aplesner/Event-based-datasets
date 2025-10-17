# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install cython numpy --break-system-packages

# Optional: for AEDAT4 support
pip install aedat --break-system-packages

# Compile the Cython module
python setup.py build_ext --inplace
```

## Basic Usage

```python
from load_aedat import load_aedat

# Load your AEDAT file
events = load_aedat('recording.aedat')

# Access the data
print(f"Loaded {len(events['x'])} events")
print(f"X coordinates: {events['x']}")      # np.int16
print(f"Y coordinates: {events['y']}")      # np.int16
print(f"Timestamps: {events['timestamp']}") # np.uint32
print(f"Polarities: {events['polarity']}")  # np.bool_
```

## All Loaders

```python
from load_aedat import load_aedat, load_aedat4, load_npy, load_binary

# AEDAT v2
events = load_aedat('file.aedat')

# AEDAT v4
events = load_aedat4('file.aedat4')

# NumPy files
events = load_npy('file.npy')

# Binary format
events = load_binary('file.bin')
```
