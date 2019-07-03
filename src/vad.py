"""Voice activity detection"""
from audio_tools import read_wav, enframe, deframe
import numpy as np


def zero_mean(framed):
    """Calculate zero-mean of frames"""
    mean = np.mean(framed, axis=1)
    framed = framed - mean[np.newaxis, :].T
    return framed


def compute_nrg(framed):
    """Energy per frame"""
    # calculate per frame energy
    n_frames = framed.shape[1]
    return np.diagonal(framed @ framed.T) / n_frames


def compute_log_nrg(framed):
    """Energy per frame in log"""
    n_frames = framed.shape[1]
    raw_nrgs = np.log(compute_nrg(framed + 1e-5)) / n_frames
    return (raw_nrgs - np.mean(raw_nrgs)) / np.std(raw_nrgs)


def vad(framed, percent_thr, nrg_thr=0, context=5):
    """
    Picks frames with high energy as determined by a 
    user defined threshold.

    This function also uses a 'context' parameter to
    resolve the fluctuative nature of thresholding. 
    context is an integer value determining the number
    of neighboring frames that should be used to decide
    if a frame is voiced.

    The log-energy values are subject to mean and var
    normalization to simplify the picking the right threshold. 
    In this framework, the default threshold is 0.0
    """
    framed = zero_mean(framed)
    n_frames = framed.shape[0]

    # Compute per frame energies:
    xnrgs = compute_log_nrg(framed)
    xvad = np.empty(n_frames)
    for i in range(n_frames):
        start = max(i - context, 0)
        end = min(i + context, n_frames - 1)
        n_above_thr = np.sum(xnrgs[start:end] > nrg_thr)
        n_total = end - start + 1
        xvad[i] = (n_above_thr / n_total) > percent_thr
    return xvad


def get_vad_ranges(file):
    """Return range of samples that contains speech"""
    rate, data = read_wav(file)
    window = rate // 40
    hop = rate // 100
    frames = enframe(data, window, hop)

    vad_output = vad(frames, 0.5)

    return _get_speech_samples(vad_output, window, hop)


def _get_speech_samples(framed, window_size, hop_size):
    """Return range of samples that contains speech"""
    ranges = []
    start = 0
    for i, out in enumerate(framed):
        if out:
            if not start:
                start = i
        else:
            if start:
                ranges.append((start * hop_size, i * hop_size + window_size))
                start = 0
    if start:
        ranges.append((start * hop_size, len(framed) * hop_size + window_size))

    return np.array(ranges)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test_file = '/home/fredy/code/ucsp/9/tcg_final/data/M_1040_11y5m_1.wav'
    rate, data = read_wav(test_file)
    window = rate // 40  # * 0.025
    hop = rate // 100  # * 0.010
    frames = enframe(data, window, hop)

    # percent_high_nrg is the VAD context ratio. It helps smooth the
    # output VAD decisions. Higher values are more strict.
    percent_high_nrg = 0.5

    vad_outup = vad(frames, percent_high_nrg)

    vad_outup = deframe(vad_outup, window, hop)

    plt.plot(data)
    plt.plot(vad_outup)
    plt.show()
