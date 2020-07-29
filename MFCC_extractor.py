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
    mfcc = MFCC(numberCoefficients=12)
    loudness = Loudness()
    spectrum = Spectrum()
    melBands = MelBands()
    onsets = OnsetDetection()
    w = Windowing(type='hann')
    c2p = CartesianToPolar() # and this turns it into a pair (magnitude, phase)
    fft = FFT() # this gives us a complex FFT

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=16384*2, hopSize=2048, startFromZero=True):
        mag, phase, = c2p(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        #average_loudness = loudness(spectrum(w(frame)))
        #mel_bands = melBands(spectrum(w(frame)))
        contrast, spectralValley = SpectralContrast()(spectrum(w(frame)))
        #print(type(contrast))
        #croma = Chromagram(sampleRate=22050)(mag[1:],)
        flatness = Flatness()(mag) 
        onset = onsets(mag,phase)
        #print((onset))
        #dynamic_complexity, loudness = dynComplex(frame(audio))
        spectral_complex = SpectralComplexity()(mag)

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        #pool.add('lowlevel.loudness', [average_loudness])
        #pool.add('lowlevel.melbands', mel_bands)
        #pool.add('lowlevel.spectralcontrast', contrast)
        #pool.add('lowlevel.chroma', croma)
        pool.add('lowlevel.flatness', [flatness])
        pool.add('lowlevel.onsets', [onset])
        pool.add('lowlevel.spectral_complexity', [spectral_complex])

    aggrPool = PoolAggregator(defaultStats=['mean'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")
    mean = json_data['lowlevel']['mfcc']['mean']
    #mel = json_data['lowlevel']['melbands']['mean']
    #loudness = json_data['lowlevel']['loudness']['mean']
    #spectContrast = json_data['lowlevel']['spectralcontrast']['mean']
    onsetdetect = json_data['lowlevel']['onsets']['mean']
    #dyncomp = json_data['lowlevel']['dynamic_complexity']['mean']

    #[[MFCC],[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],[BeatStatistics]];

    #print(mel,mean,loudness)
    #os.remove("mfccmean.json")
    return {"file": audio_file, 
            "mfccMean": mean, 
            #"mel": mel, 
            #"loudness": loudness, 
            #"spectralContrast": spectContrast,
            #"chroma": json_data['lowlevel']['chroma']['mean'],
            "flatness": json_data['lowlevel']['flatness']['mean'],
            "onsets": onsetdetect,
            "complexity": json_data['lowlevel']['spectral_complexity']['mean']
            }
    #return {"file": audio_file, "spectralContrast": spectContrast}

def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

#extrange bug: if i give less than 16 features wont write into the csv file
def concat_features(input_data):
    features = list(map(lambda data: 
               list(tz.concat(getProps(
                   ["mfccMean", "flatness", 'onsets', "complexity"], 
                   data))),
    #features = list(map(lambda data: list(tz.concat(getProps(["spectralContrast"], data))),
    input_data))
    print(features)
    return features

def save_as_matrix(features):
    save_descriptors_as_matrix('dataBaseAsMatrix_2.csv', features)

#test
#input_data = extract_all_mfccs(glob.glob('Segments_2/' + "*.wav"))

#save_as_matrix(concat_features(input_data))

# save_as_json('mfccs.json', extract_all_mfccs(
#     glob.glob('audioClases/Clase_5/' + "*.wav")))
#save_matrix_array('test.csv', 'mfccs.json')