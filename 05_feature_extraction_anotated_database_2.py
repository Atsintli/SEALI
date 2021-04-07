import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv
import os
from utils import get_json, save_descriptors_as_matrix
import json
import toolz as tz

#in_dir = 'clusters_flauta_2/'
in_dir = 'sink_into_returno/'
file_out = open('flautaAnotatedDataBase.csv', 'w') #for erasing the file if already has data
#f_out = 'anotatedMFCCsAsStrings.csv'

def extract_features(audio_file):
    loader = essentia.standard.MonoLoader(filename=audio_file)
    audio = loader()
    spectrum = Spectrum()
    melBands = MelBands()
    w = Windowing(type='hann')
    fft = FFT() # this gives us a complex FFT

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=2048, startFromZero=True):
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=13)(mag)
        #mel_bands = MelBands()(spectrum(w(frame)))
        contrast, spectralValley = SpectralContrast()(mag)
        flatness = Flatness()(mag)
        #dens = Welch()(spectrum(w(frame)))
        #onset = OnsetDetection()(mag,phase)
        #dynamic_complexity, loudness = DynamicComplexity()(mag)
        spectral_complex = SpectralComplexity()(mag)
        centroid = Centroid()(mag)
        #croma = Chromagram(sampleRate=2048*5)(mag[1:],)
        loudness = Loudness()(mag)

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', [loudness])
        #pool.add('lowlevel.melbands', mel_bands)
        pool.add('lowlevel.spectralcontrast', contrast)
        pool.add('lowlevel.flatness', [flatness])
        #pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.dyncomplex', [dynamic_complexity])
        pool.add('lowlevel.spectral_complexity', [spectral_complex])
        #pool.add('lowlevel.chroma', croma)
        #pool.add('lowlevel.dens', dens)
        pool.add('lowlevel.centroid', [centroid])
    
    #pool.add('audio_file')
    aggrPool = PoolAggregator(defaultStats=['mean','var'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")
    #dyncomp = json_data['lowlevel']['dynamic_complexity']['mean']

    #SCMIR Audio Features
    #[[MFCC],[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness]];

    #os.remove("mfccmean.json")
    return {#"file": json_data['audio_file'], 
            "mfccMean": json_data['lowlevel']['mfcc']['mean'], 
            "mfccVar": json_data['lowlevel']['mfcc']['var'], 
            #"mel": json_data['lowlevel']['melbands']['mean'], 
            "loudness": json_data['lowlevel']['loudness']['mean'],
            "spectralContrast": json_data['lowlevel']['spectralcontrast']['mean'],
            # "chroma": json_data['lowlevel']['chroma']['mean'],
            "flatness": json_data['lowlevel']['flatness']['mean'],
            #"onsets": json_data['lowlevel']['onsets']['mean'],
            #"dyncomplexity": json_data['lowlevel']['dyncomplex']['mean'],
            "complexity": json_data['lowlevel']['spectral_complexity']['mean'],
            #"dens": json_data['lowlevel']['dens']['mean'],
            #"densVar": json_data['lowlevel']['dens']['var'],
            "centroid": json_data['lowlevel']['centroid']['mean']
            }

for root, dirs, files in os.walk(in_dir):
    path = root.split(os.sep)
    class_number = root.split('_')[-1] # to obtain the number of class
    print(class_number)
    root = os.path.basename(root)+'/'
    for file in files:
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() == ".wav":
            newpath = in_dir + root + file
            extract_features(newpath)

def extract_all_mfccs(audio_files):
    return list(map(extract_features, audio_files))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data: 
               list(tz.concat(getProps(
                   #["mfccMean", "flatness", 'onsets', "complexity", "spectralContrast", "loudness"], 
                   #["mfccMean", "loudness"], 
                    ['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'], 
                    data))),
    input_data))
    print(features)
    return features
 
def save_as_matrix(features):
    features = features.append(class_number)
    save_descriptors_as_matrix('flautaAnotatedDataBase.csv', features)
    #print("INFO", type(features))

#test
print('hola')
#input_data = extract_all_mfccs(glob.glob(in_dir + "*.wav"))
#save_as_matrix(concat_features(input_data))