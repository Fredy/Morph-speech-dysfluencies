import os
import cv2
import sys
import morphological as morph
from connected_component import con_component
from mfcc import mfcc, fbank as fbank_or
from audio_tools import read_wav, enframe
from vad import vad, _get_speech_frames, _get_speech_samples
from similarity import self_similarity, cross_correlation, distance
from os import path
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np


def voice_activity_det(wav_file):
    """
    First step: Voice activity detection.

    :param wav_file: File name.
    :return: Audio frames, List of frames' ranges of detected voice and sample rate. 
    """

    rate, data = read_wav(wav_file)

    window = int(rate * 0.030)
    hop = int(rate * 0.020)

    frames = enframe(data, window, hop).astype('float32')
    vad_out = vad(frames, 0.2, context=15)
    vad_samples = _get_speech_samples(vad_out, window, hop)

    for i, (start, end) in enumerate(vad_samples):
        file_name = f'res/{path.basename(wav_file)}/{i:04}.wav'
        wavfile.write(file_name, rate, data[start:end])

    ranges = _get_speech_frames(vad_out)
    return frames, ranges, rate


def fbank(frames, samplerate):
    return fbank_or(frames, samplerate)[0]


def similarities(frames, ranges, rate, feature_ext=mfcc,
                 measure=cross_correlation, normalize=False):
    """
    Extract features and find self similarity of frames

    :param frames: Voice frames
    :param ranges: Range of detected voice in frames
    :param rate: Audio sample rate
    :param feature_ext: Feature extractor, defaults to mfcc
    :param measure: Measure to fint the difference between features, defaults
                    to cross_correlation
    :return: List of self similarity matrices
    """
    imgs = []
    for i, (start, end) in enumerate(ranges):
        features = feature_ext(frames[start:end], rate)
        similarities = self_similarity(features, measure, normalize)

        imgs.append(similarities)

        file_name = (f'res/{path.basename(wav_file)}/'
                     f'{i:04}-{feature_ext.__name__}'
                     f'-{measure.__name__}.png')
        cv2.imwrite(file_name, similarities * 255)

    return imgs


def _plot_img(idx, img):
    plt.subplots_adjust(0, 0, 1, 1, 0.01)
    plt.subplot(idx)
    plt.axis('off')
    plt.imshow(img)


class Detector:

    @staticmethod
    def prolongation(matrix, threshold_bin=0.88, cl=(3, 3), op=(20, 20), idx=None):
        _space = np.ones((matrix.shape[0], 6)) * 255
        tmp = matrix * 255
        matrix = matrix > threshold_bin
        tmp = np.concatenate((tmp, _space, matrix * 255), axis=1)
        labels = con_component(matrix, max_label=1)
        matrix = labels.astype(bool)  # Just get the diagonal
        tmp = np.concatenate((tmp, _space, matrix * 255), axis=1)

        matrix = morph.closing(matrix, np.ones(cl, bool))
        tmp = np.concatenate((tmp, _space, matrix * 255), axis=1)
        matrix = morph.opening(matrix, np.ones(op, bool))
        tmp = np.concatenate((tmp, _space, matrix * 255), axis=1)

        cv2.imwrite(f'res/{idx or 0}-prolongation.png', tmp)
        return matrix

    @staticmethod
    def word_repetition(matrix, threshold_bin=0.83, op=(3, 3), idx=None):
        """Word and syllabe repetition"""
        _space = np.ones((matrix.shape[0], 6)) * 255
        tmp = matrix * 255
        matrix = matrix > threshold_bin
        tmp = np.concatenate((tmp, _space, matrix * 255), axis=1)
        labels = con_component(matrix, max_label=1)

        matrix = np.where(labels, np.zeros_like(matrix), matrix)
        tmp = np.concatenate((tmp, _space, matrix * 255), axis=1)

        matrix = morph.opening(matrix, np.ones(op, bool))
        tmp = np.concatenate((tmp, _space, matrix * 255), axis=1)

        cv2.imwrite(f'res/{idx or 0}-repetition_wp.png', tmp)
        return matrix


if __name__ == "__main__":
    wav_file = ''
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
    else:
        print('Please give a valid wav file name')
        exit(1)

    os.makedirs(f'res/{path.basename(wav_file)}', exist_ok=True)

    frames, ranges, rate = voice_activity_det(wav_file)

    matrices = similarities(frames, ranges, rate, feature_ext=mfcc,
                            measure=cross_correlation)
    # - Distance:
    #   use normalize
    #   prolongation threshold: 0.73
    #   repetition threshold: 0.68
    # prolongation = Detector.prolongation(img, threshold_bin=0.73, idx=idx)
    # repetition = Detector.word_repetition(img, threshold_bin=0.68, idx=idx)

    # - Cross correlation:
    #   not normalize
    #   all thresholds in default

    for idx, img in enumerate(matrices):
        prolongation = Detector.prolongation(img, idx=idx)
        repetition = Detector.word_repetition(img, idx=idx)

        if prolongation.any():
            # remove the bottom half (diagonal) of the matrix
            # the top half has the needed data
            for i in range(len(prolongation)):
                prolongation[i, :i] = 0
            bbox = cv2.boundingRect((prolongation * 255).astype('uint8'))
            # TODO: this should be extracted from vad!
            window = int(rate * 0.030)
            hop = int(rate * 0.020)
            start = bbox[0] * hop / rate
            end = ((bbox[0] + bbox[3]) * hop + window) / rate
            print(f'{idx:03}: Prolongation: {start:05.2f} - {end:05.2f} {bbox}')

        if repetition.any():
            # remove the bottom half (diagonal) of the matrix
            # the top half has the needed data
            for i in range(len(repetition)):
                repetition[i, :i] = 0
            bbox = cv2.boundingRect((repetition * 255).astype('uint8'))
            if bbox[2] < 15 and bbox[3] < 15:
                continue
            # TODO: this should be extracted from vad!
            window = int(rate * 0.030)
            hop = int(rate * 0.020)
            start = bbox[0] * hop / rate
            end = ((bbox[0] + bbox[3]) * hop + window) / rate
            print(f'{idx:03} Repetition   : {start:05.2f} - {end:05.2f} {bbox}')
