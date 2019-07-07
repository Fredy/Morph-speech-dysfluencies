import os
import cv2
import sys
import morphological as morph
from connected_component import con_component
from mfcc import mfcc
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


def similarities(frames, ranges, rate, feature_ext=mfcc,
                 measure=cross_correlation):
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
        similarities = self_similarity(features, measure)

        imgs.append(similarities)

        file_name = (f'res/{path.basename(wav_file)}/'
                     f'{i:04}-{feature_ext.__name__}'
                     f'-{measure.__name__}.png')
        cv2.imwrite(file_name, similarities * 255)

    return imgs


def _plot_img(idx, img):
    plt.subplot(idx)
    plt.axis('off')
    plt.imshow(img)


class Detector:
    @staticmethod
    def prolongation(matrix, threshold_bin=0.88, cl=(3, 3), op=(20, 20)):
        _plot_img(151, matrix)
        matrix = matrix > threshold_bin
        _plot_img(152, matrix)
        labels = con_component(matrix)
        matrix = labels == 1  # Just get the diagonal
        _plot_img(153, matrix)

        matrix = morph.closing(matrix, np.ones(cl, bool))
        _plot_img(154, matrix)
        matrix = morph.opening(matrix, np.ones(op, bool))
        _plot_img(155, matrix)

        plt.savefig('algo.png')
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
