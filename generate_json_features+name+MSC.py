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

    name = audio_file.split('/')[1].split('.')[-2]

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=2048, startFromZero=True): #for chroma frameSize=8192*2, hopSize=8192, #fz=88200, hs=44100
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=13)(mag)
        loudness = Loudness()(mag)

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        #pool.add('lowlevel.loudness', [loudness])

    aggrPool = PoolAggregator(defaultStats=['mean', 'var'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")

    return {"file": name, 
            "mfccMean": json_data['lowlevel']['mfcc']['mean'],
            "mfccVar": json_data['lowlevel']['mfcc']['var']
            #"loudness": json_data['lowlevel']['loudness']['mean']
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
        f.write(json.dumps(data, sort_keys=False, indent=1, separators=(',', ': '))) #for pretty

def add_MSC_to_files(audio_files):
    mfcc_data = extract_all_mfccs(audio_files)
    features = list( map(lambda file_data: file_data["mfccMean"], mfcc_data))
    features = list( map(lambda file_data: file_data["mfccVar"], mfcc_data))
    #features = list( map(lambda file_data: file_data["loudness"], mfcc_data))
    labels = meanShift(features)[1]
    return [tz.assoc(file_data, "className", str(labels[i])) for i,file_data in enumerate(mfcc_data)] 

#run all
input_data = sorted(glob.glob('segments_short/' + "*.wav"))
classified_data = add_MSC_to_files(input_data)
grouped_data = tz.groupby(lambda data: data["className"], classified_data)
convert_write_json(grouped_data, 'anotated_mfcc_mean_var.json')
