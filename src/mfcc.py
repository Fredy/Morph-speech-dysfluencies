"""
Mel Frequency Cepstral Co-efficients
Easy explanation: https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd
http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
"""
import python_speech_features
from scipy.fftpack import dct
import numpy as np


def mfcc(frames, samplerate):
    features, _ = fbank(frames, samplerate)
    features = np.log(features)
    # just 12 coefficients
    features = dct(features, axis=1, norm='ortho')[:, 1:13]
    features = lifter(features)

    return features


def fbank(frames, samplerate):
    highfreq = samplerate/2
    # TODO: apply this filter before framming
    for i in range(len(frames)):
        frames[i] = preemphasis(frames[i])

    power_spec = powspec(frames, 512)
    energy = np.sum(power_spec)
    # if energy is zero, we get problems with log
    energy = np.where(energy == 0, np.finfo(float).eps, energy)

    filter_banks = get_filterbanks(26, 512, samplerate, 0, highfreq)
    features = np.dot(power_spec, filter_banks.T)

    # if feat is zero, we get problems with log
    features = np.where(features == 0, np.finfo(float).eps, features)

    return features, energy


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz_to_mel(lowfreq)
    highmel = hz_to_mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel_to_hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt, nfft//2+1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j, i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j, i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank


def hz_to_mel(hz):
    return 2595 * np.log10(1+hz/700.)


def mel_to_hz(mel):
    return 700*(10**(mel/2595.0)-1)


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """

    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)
