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

in_dir = 'audioClases/'
file_out = open('anotatedDataBase.csv', 'w') #for erasing the file if already has data
#f_out = 'anotatedMFCCsAsStrings.csv'

def extract_features(audio_file):
    loader = essentia.standard.MonoLoader(filename=audio_file)
    audio = loader()
    spectrum = Spectrum()
    melBands = MelBands()
    w = Windowing(type='hann')
    fft = FFT() # this gives us a complex FFT

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=88200, hopSize=44100, startFromZero=True):
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=1)(mag)
        print (mfcc_bands)
        loudness = Loudness()(mag)
        #mel_bands = melBands(spectrum(w(frame)))
        #contrast, spectralValley = SpectralContrast()(mag)
        #croma = Chromagram(sampleRate=22050)(mag[1:],)
        #flatness = Flatness()(mag) 
        #onset = OnsetDetection()(mag,phase)
        #print((onset))
        #dynamic_complexity, loudness = dynComplex(frame(audio))
        #spectral_complex = SpectralComplexity()(mag)

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', [loudness])
        #pool.add('lowlevel.melbands', mel_bands)
        #pool.add('lowlevel.spectralcontrast', contrast)
        #pool.add('lowlevel.chroma', croma)
        #pool.add('lowlevel.flatness', [flatness])
        #pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.spectral_complexity', [spectral_complex])

    aggrPool = PoolAggregator(defaultStats=['mean'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")
    #dyncomp = json_data['lowlevel']['dynamic_complexity']['mean']

    #[[MFCC],[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],[BeatStatistics]];

    #os.remove("mfccmean.json")
    return {"file": audio_file, 
            "mfccMean": json_data['lowlevel']['mfcc']['mean'], 
            #"mel": json_data['lowlevel']['melbands']['mean'], 
            "loudness": json_data['lowlevel']['loudness']['mean'],
            #"spectralContrast": json_data['lowlevel']['spectralcontrast']['mean'],
            #"chroma": json_data['lowlevel']['chroma']['mean'],
            #"flatness": json_data['lowlevel']['flatness']['mean'],
            #"onsets": json_data['lowlevel']['onsets']['mean'],
            #"complexity": json_data['lowlevel']['spectral_complexity']['mean']
            }

for root, dirs, files in os.walk(in_dir):
    path = root.split(os.sep)
    class_number = root.split('_')[-1] # to obtain the number of class
    #print(class_number)
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
                   ["mfccMean", "loudness"], 
                   data))),
    input_data))
    print(features)
    return features
 
def save_as_matrix(features):
    features = features.append(class_number)
    save_descriptors_as_matrix('anotatedDataBase.csv', features)
    #print("INFO", type(features))

#test
#input_data = extract_all_mfccs(glob.glob('audioClases/' + "*.wav"))
#save_as_matrix(concat_features(input_data))