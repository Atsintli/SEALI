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
    print("Analyzing:" + audio_file)
    audio = loader()
    #spectrum = Spectrum()
    #melBands = MelBands()
    w = Windowing(type='hann')
    fft = FFT()

    name = audio_file.split('/')[1].split('.')[-2]
    print(name)

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=2048, startFromZero=True): #for chroma frameSize=8192*2, hopSize=8192, #fz=88200, hs=44100
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=13)(mag)
        #mel_bands = melBands(spectrum(w(frame)))
        # contrast, spectralValley = SpectralContrast()(mag)
        #flatness = Flatness()(mag) 
        #onset = OnsetDetection()(mag,phase)
        #dynamic_complexity, loudness = DynamicComplexity()(mag)
        #spectral_complex = SpectralComplexity()(mag)
        #croma = Chromagram(sampleRate=2048*5)(mag[1:],)
        #loudness = Loudness()(mag)

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        #pool.add('lowlevel.loudness', [loudness])
        #pool.add('lowlevel.melbands', mel_bands)
        #pool.add('lowlevel.spectralcontrast', contrast)
        #pool.add('lowlevel.flatness', [flatness])
        #pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.dyncomplex', [dynamic_complexity])
        #pool.add('lowlevel.spectral_complexity', [spectral_complex])
        #pool.add('lowlevel.chroma', croma)
    
    pool.add('audio_file', (name))
    aggrPool = PoolAggregator(defaultStats=['mean','var'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")
    #dyncomp = json_data['lowlevel']['dynamic_complexity']['mean']

    #SCMIR Audio Features
    #[[MFCC],[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness]];

    #os.remove("mfccmean.json")
    return {"file": json_data['audio_file'], 
            "mfccMean": json_data['lowlevel']['mfcc']['mean'], 
            "mfccVar": json_data['lowlevel']['mfcc']['var'], 
            #"mel": json_data['lowlevel']['melbands']['mean'], 
            #"loudness": json_data['lowlevel']['loudness']['mean'],
            # "spectralContrast": json_data['lowlevel']['spectralcontrast']['mean'],
            # "chroma": json_data['lowlevel']['chroma']['mean'],
            #"flatness": json_data['lowlevel']['flatness']['mean'],
            #"onsets": json_data['lowlevel']['onsets']['mean'],
            #"dyncomplexity": json_data['lowlevel']['dyncomplex']['mean'],
            #"complexity": json_data['lowlevel']['spectral_complexity']['mean']
            }

def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data: 
               list(tz.concat(getProps(
                   #['flatness', 'complexity', 'dyncomplexity','mfccMean','onsets'], 
                   #['mfccMean','flatness', 'complexity', 'onsets'],
                   #['mel'],
                   ['mfccMean', 'mfccVar'],
                   #['loudness'],
                   data))),
    input_data))
    #print(features)
    return features

def save_as_matrix(features):
    save_descriptors_as_matrix('for_than_foam_mfcc_mean_variance_.csv', features)

#test

input_data = extract_all_mfccs(sorted(glob.glob('segments_short/' + "*.wav")))
#print(input_data)
save_as_matrix(concat_features(input_data))
