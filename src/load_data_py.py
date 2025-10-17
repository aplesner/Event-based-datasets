import aedat  # type: ignore
import numpy as np
import struct

def load_aedat(filepath: str) -> dict[str, np.ndarray]:
    """
    Load events from an AEDAT file (version 2).
    
    Args:
        filepath: Path to the .aedat file
        
    Returns:
        Dictionary containing 'x', 'y', 'timestamp', and 'polarity' arrays
    """
    with open(filepath, 'rb') as f:
        # Skip header lines starting with '#'
        while True:
            position = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(position)
                break
        # Read binary event data
        events = []
        while True:
            data = f.read(8)  # Each event is 8 bytes
            if len(data) < 8:
                break
            
            # Unpack event data
            addr, timestamp = struct.unpack('II', data)
            
            # Extract x, y, polarity from address
            x = (addr >> 12) & 0x3FF
            y = (addr >> 22) & 0x3FF
            polarity = (addr >> 11) & 0x1
            
            events.append([x, y, timestamp, polarity])
    
    if not events:
        return {
            'x': np.array([]),
            'y': np.array([]),
            'timestamp': np.array([]),
            'polarity': np.array([])
        }
    
    events_array = np.array(events)
    return {
        'x': events_array[:, 0].astype(np.int16),
        'y': events_array[:, 1].astype(np.int16),
        'timestamp': events_array[:, 2].astype(np.uint32),
        'polarity': events_array[:, 3].astype(np.bool_)
    }

def load_aedat4(filepath: str) -> dict[str, np.ndarray]:
    """
    Load events from an AEDAT4 file.
    
    Args:
        filepath: Path to the .aedat4 file
    Returns:
        Dictionary containing 'x', 'y', 'timestamp', and 'polarity' arrays
    """
    decoder = aedat.Decoder(filepath)  # type: ignore
    target_id = None
    for stream_id, stream in decoder.id_to_stream().items():
        if stream["type"] == "events" and (target_id is None or stream_id < target_id):
            target_id = stream_id
    if target_id is None:
        raise Exception("there are no events in the AEDAT file")
    events = np.concatenate(tuple(
                packet["events"]
                for packet in decoder
                if packet["stream_id"] == target_id
            ))
    x = np.array([e[1] for e in events], dtype=np.int16)
    y = np.array([e[2] for e in events], dtype=np.int16)
    t = np.array([e[0] for e in events], dtype=np.uint32)
    p = np.array([e[3] for e in events], dtype=np.bool_)
    return {
        'x': x,
        'y': y,
        'timestamp': t,
        'polarity': p
    }
   

def load_npy(filepath: str) -> dict[str, np.ndarray]:
    """
    Load events from a .npy file.
    
    Args:
        filepath: Path to the .npy file
    Returns:
        Dictionary containing 'x', 'y', 'timestamp', and 'polarity' arrays
    """
    events = np.load(filepath)
    x = np.array([e[1] for e in events], dtype=np.int16)
    y = np.array([e[2] for e in events], dtype=np.int16)
    t = np.array([e[0] for e in events], dtype=np.uint32)
    p = np.array([e[3] for e in events], dtype=np.bool_)
    return {
        'x': x,
        'y': y,
        'timestamp': t,
        'polarity': p
    }

def load_binary(filepath: str) -> dict[str, np.ndarray]:
    """
    Load events from a custom binary file.
    Code from the tonic library and Garrick Orchard (https://github.com/gorchard/event-Python/blob/master/eventvision.py#L532-L560).
    Adapted to return a dictionary.
    
    Args:
        filepath: Path to the binary file
    Returns:
        Dictionary containing 'x', 'y', 'timestamp', and 'polarity' arrays
    """
    with open(filepath, "rb") as fp:
            raw_data = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)

    all_x = raw_data[0::5]
    all_y = raw_data[1::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2**13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    return {
        'x': all_x[td_indices].astype(np.int16),
        'y': all_y[td_indices].astype(np.int16),
        'timestamp': all_ts[td_indices].astype(np.uint32),
        'polarity': all_p[td_indices].astype(np.bool_)
    }