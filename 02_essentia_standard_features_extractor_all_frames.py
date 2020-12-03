import json
import essentia
import essentia.standard
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv
import os

spectrum = Spectrum()
w = Windowing(type='hann')
fft = FFT()
mfcc = MFCC() 

def extract_mfccs(audio_file):
    print("Analyzing:" + audio_file)
    loader = essentia.standard.MonoLoader(filename=audio_file)
    audio = loader()
    frame = audio[1*44100 : 1*44100 + 512]
    spec = spectrum(w(frame))
    mfcc_bands, mfcc_coeffs = mfcc(spec)

    mfccs = []

    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))

        mfccs.append(mfcc_coeffs)
    return mfccs

def save_as_matrix(features):
    save_matrix_array('test_mfccs.csv', features)

def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

input_data = extract_all_mfccs(sorted(glob.glob('audios_test/' + "*.wav")))
save_as_matrix(input_data)