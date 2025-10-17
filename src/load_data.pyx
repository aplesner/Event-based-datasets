# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdio cimport FILE, fopen, fclose, fread, fseek, ftell, SEEK_SET, SEEK_END
from libc.stdint cimport uint32_t, int16_t, uint8_t

cnp.import_array()

def load_aedat(filepath: str) -> dict[str, cnp.ndarray]:
    """
    Load events from an AEDAT file (version 2).
    
    Args:
        filepath: Path to the .aedat file
        
    Returns:
        Dictionary containing 'x', 'y', 'timestamp', and 'polarity' arrays
    """
    cdef:
        FILE* file_ptr
        bytes filepath_bytes = filepath.encode('utf-8')
        char* filepath_c = filepath_bytes
        unsigned char buffer[8]
        long data_start_position, file_size
        uint32_t x, y, polarity, addr, timestamp
        size_t bytes_read
        long long num_events, i
        cnp.ndarray[int16_t, ndim=1] x_arr
        cnp.ndarray[int16_t, ndim=1] y_arr
        cnp.ndarray[uint32_t, ndim=1] timestamp_arr
        cnp.ndarray[uint8_t, ndim=1] polarity_arr
        int16_t[:] x_view
        int16_t[:] y_view
        uint32_t[:] timestamp_view
        uint8_t[:] polarity_view
    
    # Use Python file I/O to skip header (matches original exactly)
    with open(filepath, 'rb') as f:
        while True:
            position = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(position)
                break
        data_start_position = f.tell()
    
    # Open file with C for fast event reading
    file_ptr = fopen(filepath_c, "rb")
    if file_ptr == NULL:
        raise IOError(f"Cannot open file: {filepath}")
    
    try:
        # Seek to where data starts
        fseek(file_ptr, data_start_position, SEEK_SET)
        
        # Get file size to estimate number of events
        fseek(file_ptr, 0, SEEK_END)
        file_size = ftell(file_ptr)
        fseek(file_ptr, data_start_position, SEEK_SET)
        
        num_events = (file_size - data_start_position) // 8
        
        if num_events <= 0:
            fclose(file_ptr)
            return {
                'x': np.array([], dtype=np.int16),
                'y': np.array([], dtype=np.int16),
                'timestamp': np.array([], dtype=np.uint32),
                'polarity': np.array([], dtype=np.bool_)
            }
        
        # Pre-allocate arrays
        x_arr = np.empty(num_events, dtype=np.int16)
        y_arr = np.empty(num_events, dtype=np.int16)
        timestamp_arr = np.empty(num_events, dtype=np.uint32)
        polarity_arr = np.empty(num_events, dtype=np.uint8)
        
        # Get memoryviews for fast access
        x_view = x_arr
        y_view = y_arr
        timestamp_view = timestamp_arr
        polarity_view = polarity_arr
        
        # Read events
        i = 0
        while True:
            bytes_read = fread(buffer, 1, 8, file_ptr)
            if bytes_read < 8:
                break
            
            # Unpack as little-endian unsigned ints (matching struct.unpack('II'))
            addr = (<uint32_t>buffer[0] | 
                   (<uint32_t>buffer[1] << 8) | 
                   (<uint32_t>buffer[2] << 16) | 
                   (<uint32_t>buffer[3] << 24))
            timestamp = (<uint32_t>buffer[4] | 
                        (<uint32_t>buffer[5] << 8) | 
                        (<uint32_t>buffer[6] << 16) | 
                        (<uint32_t>buffer[7] << 24))
            
            # Extract x, y, polarity from address
            x = (addr >> 12) & 0x3FF
            y = (addr >> 22) & 0x3FF
            polarity = (addr >> 11) & 0x1
            
            x_view[i] = <int16_t>x
            y_view[i] = <int16_t>y
            timestamp_view[i] = timestamp
            polarity_view[i] = <uint8_t>polarity
            
            i += 1
        
        # Trim arrays if we read fewer events than expected
        if i < num_events:
            x_arr = x_arr[:i]
            y_arr = y_arr[:i]
            timestamp_arr = timestamp_arr[:i]
            polarity_arr = polarity_arr[:i]
        
    finally:
        fclose(file_ptr)
    
    return {
        'x': x_arr,
        'y': y_arr,
        'timestamp': timestamp_arr,
        'polarity': polarity_arr.astype(np.bool_)
    }


def load_aedat4(filepath: str) -> dict[str, np.ndarray]:
    """
    Load events from an AEDAT4 file.
    
    Args:
        filepath: Path to the .aedat4 file
    Returns:
        Dictionary containing 'x', 'y', 'timestamp', and 'polarity' arrays
    """
    import aedat
    
    decoder = aedat.Decoder(filepath)  # type: ignore
    target_id = None
    for stream_id, stream in decoder.id_to_stream().items():
        if stream["type"] == "events" and (target_id is None or stream_id < target_id):
            target_id = stream_id
    if target_id is None:
        raise Exception("there are no events in the AEDAT file")
    
    # Collect all event packets
    events = np.concatenate(tuple(
                packet["events"]
                for packet in decoder
                if packet["stream_id"] == target_id
            ))
    
    # Use direct array slicing instead of list comprehensions
    # This is much faster than iterating through events
    if events.ndim == 2:
        # Events are structured as (t, x, y, p) columns
        return {
            'x': events[:, 1].astype(np.int16),
            'y': events[:, 2].astype(np.int16),
            'timestamp': events[:, 0].astype(np.uint32),
            'polarity': events[:, 3].astype(np.bool_)
        }
    else:
        # Fallback to element access if structure is different
        return _extract_aedat4_fields(events)


def _extract_aedat4_fields(events: np.ndarray) -> dict[str, np.ndarray]:
    """
    Extract fields from AEDAT4 events array using fast Cython loops.
    """
    cdef:
        long long n = len(events)
        long long i
        cnp.ndarray[int16_t, ndim=1] x_arr = np.empty(n, dtype=np.int16)
        cnp.ndarray[int16_t, ndim=1] y_arr = np.empty(n, dtype=np.int16)
        cnp.ndarray[uint32_t, ndim=1] t_arr = np.empty(n, dtype=np.uint32)
        cnp.ndarray[uint8_t, ndim=1] p_arr = np.empty(n, dtype=np.uint8)
        int16_t[:] x_view = x_arr
        int16_t[:] y_view = y_arr
        uint32_t[:] t_view = t_arr
        uint8_t[:] p_view = p_arr
    
    for i in range(n):
        event = events[i]
        t_view[i] = <uint32_t>event[0]
        x_view[i] = <int16_t>event[1]
        y_view[i] = <int16_t>event[2]
        p_view[i] = <uint8_t>event[3]
    
    return {
        'x': x_arr,
        'y': y_arr,
        'timestamp': t_arr,
        'polarity': p_arr.astype(np.bool_)
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
    
    # Use direct array slicing instead of list comprehensions (much faster)
    if events.ndim == 2 and events.shape[1] >= 4:
        # Events are structured as columns: (t, x, y, p) or similar
        return {
            'x': events[:, 1].astype(np.int16),
            'y': events[:, 2].astype(np.int16),
            'timestamp': events[:, 0].astype(np.uint32),
            'polarity': events[:, 3].astype(np.bool_)
        }
    else:
        # Fallback to element access if structure is different
        return _extract_npy_fields(events)


def _extract_npy_fields(events: np.ndarray) -> dict[str, np.ndarray]:
    """
    Extract fields from npy events array using fast Cython loops.
    """
    cdef:
        long long n = len(events)
        long long i
        cnp.ndarray[int16_t, ndim=1] x_arr = np.empty(n, dtype=np.int16)
        cnp.ndarray[int16_t, ndim=1] y_arr = np.empty(n, dtype=np.int16)
        cnp.ndarray[uint32_t, ndim=1] t_arr = np.empty(n, dtype=np.uint32)
        cnp.ndarray[uint8_t, ndim=1] p_arr = np.empty(n, dtype=np.uint8)
        int16_t[:] x_view = x_arr
        int16_t[:] y_view = y_arr
        uint32_t[:] t_view = t_arr
        uint8_t[:] p_view = p_arr
    
    for i in range(n):
        event = events[i]
        t_view[i] = <uint32_t>event[0]
        x_view[i] = <int16_t>event[1]
        y_view[i] = <int16_t>event[2]
        p_view[i] = <uint8_t>event[3]
    
    return {
        'x': x_arr,
        'y': y_arr,
        'timestamp': t_arr,
        'polarity': p_arr.astype(np.bool_)
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
    
    # Process time stamp overflow events using fast Cython
    all_ts = _process_overflows(all_ts, all_y)
    
    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]
    return {
        'x': all_x[td_indices].astype(np.int16),
        'y': all_y[td_indices].astype(np.int16),
        'timestamp': all_ts[td_indices].astype(np.uint32),
        'polarity': all_p[td_indices].astype(np.bool_)
    }


def _process_overflows(all_ts: np.ndarray, all_y: np.ndarray) -> np.ndarray:
    """
    Process timestamp overflow events efficiently using Cython.
    """
    cdef:
        cnp.ndarray[uint32_t, ndim=1] ts_arr = all_ts.astype(np.uint32)
        cnp.ndarray[uint32_t, ndim=1] y_arr = all_y.astype(np.uint32)
        uint32_t[:] ts_view = ts_arr
        uint32_t[:] y_view = y_arr
        long long n = len(all_ts)
        long long i
        uint32_t time_increment = 2**13
        uint32_t cumulative_increment = 0
    
    # Single pass: accumulate increments and apply them
    for i in range(n):
        if y_view[i] == 240:
            cumulative_increment += time_increment
        ts_view[i] += cumulative_increment
    
    return ts_arr
