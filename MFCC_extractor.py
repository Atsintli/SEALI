import json
import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv
import os
from utils import get_json, save_as_json, save_matrix_array
from utils import save_descriptors_as_matrix 
import toolz as tz


def extract_mfccs(audio_file):
    loader = essentia.standard.MonoLoader(filename=audio_file)
    audio = loader()
    mfcc = MFCC(numberCoefficients=13)
    loudness = Loudness()
    spectrum = Spectrum()
    melBands = MelBands()
    w = Windowing(type='hann')

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        average_loudness = loudness(spectrum(w(frame)))
        mel_bands = melBands(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', average_loudness)
        pool.add('lowlevel.melbands', mel_bands)
    aggrPool = PoolAggregator(defaultStats=['mean'])(pool)
    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)
    
    json_data = get_json("features.json")
    mean = json_data['lowlevel']['mfcc']['mean']
    mel = json_data['lowlevel']['melbands']['mean']
    loudness = json_data['lowlevel']['loudness']['mean']
    print(mel,mean,loudness)
    #os.remove("mfccmean.json")
    return {"file": audio_file, "mean": mean, "mel": mel, "loudness": [loudness]}

def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

# test

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features():
    features = list(map(lambda data: list(tz.concat(getProps(["mel","mean","loudness"], data))),
    extract_all_mfccs(glob.glob('Segments/' + "*.wav"))))
    print(features)
    return features

def save_as_matrix():
    save_descriptors_as_matrix('dataBaseAsMatrix.csv', concat_features())

save_as_matrix()


#save_descriptors_as_matrix('dataBaseAsMatrix.csv', features)


#save_as_matrix()

# save_as_json('mfccs.json', extract_all_mfccs(
#     glob.glob('audioClases/Clase_5/' + "*.wav")))

#save_matrix_array('test.csv', 'mfccs.json')