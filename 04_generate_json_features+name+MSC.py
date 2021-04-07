import json
import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
import os
from utils import get_json, save_as_json, save_matrix_array
from utils import save_descriptors_as_matrix 
import toolz as tz
from sklearn.cluster import MeanShift, estimate_bandwidth

def extract_mfccs(audio_file):
    loader = essentia.standard.MonoLoader(filename=audio_file)
    print("Analyzing:" + audio_file)
    audio = loader()
    spectrum = Spectrum()
    melBands = MelBands()
    w = Windowing(type='hann')
    fft = FFT()

    #name = audio_file.split('/')[1].split('.')[-2]
    name = audio_file.split('.')[-2].split('/')[-1]

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=2048, startFromZero=True): #for chroma frameSize=8192*2, hopSize=8192, #fz=88200, hs=44100
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=13)(mag)
        contrast, spectralValley = SpectralContrast()(mag)
        flatness = Flatness()(mag)
        spectral_complex = SpectralComplexity()(mag)
        centroid = Centroid()(mag)
        loudness = Loudness()(mag)
        #dens = Welch()(spectrum(w(frame)))
        #onset = OnsetDetection()(mag,phase)
        #dynamic_complexity, loudness = DynamicComplexity()(mag)
        #croma = Chromagram(sampleRate=2048*5)(mag[1:],)

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', [loudness])
        pool.add('lowlevel.spectralcontrast', contrast)
        pool.add('lowlevel.flatness', [flatness])
        pool.add('lowlevel.spectral_complexity', [spectral_complex])
        pool.add('lowlevel.centroid', [centroid])
        #pool.add('lowlevel.dyncomplex', [dynamic_complexity])
        #pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.chroma', croma)
        #pool.add('lowlevel.dens', dens)

    aggrPool = PoolAggregator(defaultStats=['mean', 'var'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")

    return {"file": name, 
            "mfccMean": json_data['lowlevel']['mfcc']['mean'], 
            "mfccVar": json_data['lowlevel']['mfcc']['var'], 
            "loudness": json_data['lowlevel']['loudness']['mean'],
            "spectralContrast": json_data['lowlevel']['spectralcontrast']['mean'],
            "flatness": json_data['lowlevel']['flatness']['mean'],
            "complexity": json_data['lowlevel']['spectral_complexity']['mean'],
            "centroid": json_data['lowlevel']['centroid']['mean']
            }

def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

def meanShift(features): # features are [[floats]]

    bandwidth = estimate_bandwidth(features, quantile=0.5, 
                                    n_samples=len(features))
    ms = MeanShift(
                bandwidth=bandwidth, 
                bin_seeding=False,
                max_iter=500,
                cluster_all=True)
    ms.fit(features)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    P = ms.predict(features)
    return n_clusters_, labels, cluster_centers, features

# Function to convert a CSV to JSON 
def convert_write_json(data, json_file):
    with open(json_file, "w") as f:
        f.write(json.dumps(data, sort_keys=False, indent=1, separators=(',', ': '))) #for prettyfy

#['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'], 
# checar que solo entra el ultimo feature a meanShift

def add_MSC_to_files(audio_files):
    mfcc_data = extract_all_mfccs(audio_files)
    features = list( map(lambda file_data: file_data["flatness"], mfcc_data))
    features = list( map(lambda file_data: file_data["mfccVar"], mfcc_data))
    features = list( map(lambda file_data: file_data["complexity"], mfcc_data))
    features = list( map(lambda file_data: file_data["mfccMean"], mfcc_data))
    features = list( map(lambda file_data: file_data["loudness"], mfcc_data))
    features = list( map(lambda file_data: file_data["centroid"], mfcc_data))
    features = list( map(lambda file_data: file_data["spectralContrast"], mfcc_data))
    labels = meanShift(features)[1]
    return [tz.assoc(file_data, "className", str(labels[i])) for i,file_data in enumerate(mfcc_data)] 

#run all
input_data = sorted(glob.glob('audiotestsplits/' + "*.wav")[0:10])
classified_data = add_MSC_to_files(input_data)
grouped_data = tz.groupby(lambda data: data["className"], classified_data)
convert_write_json(grouped_data, 'audiotestsplits_2.json')
