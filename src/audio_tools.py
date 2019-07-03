"""General audio tools"""
import numpy as np
from scipy.io import wavfile


def read_wav(file):
    """Read and normalizes wavfile between [-1, 1]"""
    rate, data = wavfile.read(file)
    if data.dtype == np.float32:
        return rate, data

    max_ = max(data.min(), data.max(), key=abs)
    data = data / max_

    return rate, data


def enframe(data, window_size, hop_size):
    """Divide 1D data in frames"""
    if data.ndim != 1:
        raise TypeError('Input must be a 1-dimensional array.')
    n_frames = 1 + (len(data) - window_size) // hop_size
    framed = np.empty((n_frames, window_size))
    for i in range(n_frames):
        framed[i] = data[i * hop_size: i * hop_size + window_size]
    return framed


def deframe(framed, window_size, hop_size):
    """Convert framed data in 1D data"""
    n_frames = len(framed)
    n_samples = n_frames * hop_size + window_size
    samples = np.empty(n_samples)
    for i in range(n_frames):
        samples[i * hop_size: i * hop_size + window_size] = framed[i]
    return samples
