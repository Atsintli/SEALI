import json
import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv
from utils import get_json, save_as_json


def extract_mfccs(audio_file):
    loader = essentia.standard.MonoLoader(filename=audio_file)
    audio = loader()
    mfcc = MFCC(numberCoefficients=13)
    spectrum = Spectrum()
    w = Windowing(type='hann')

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
    aggrPool = PoolAggregator(defaultStats=['mean'])(pool)
    YamlOutput(filename='mfccmean.json', format='json',
               writeVersion=False)(aggrPool)
    mean = get_json("mfccmean.json")['lowlevel']['mfcc']['mean']
    return {"filename": audio_file, "mean": mean}


def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

# test


save_as_json('mfccs.json', extract_all_mfccs(
    glob.glob('Segments/' + "*.wav")))
