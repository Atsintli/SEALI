import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv
import os
from utils import get_json
import json

in_dir = 'clusters_sinkintoreturn_2_PCA/'
csv_file = 'anotated_dataset_sinkintoreturn.csv'
file_out = open(csv_file, 'w') #for erasing the file if already has data
#f_out = 'anotatedDataBase_movil.csv'
#f_out = 'anotatedMFCCsAsStrings.csv'

def extract_features(path):
  loader = essentia.standard.EqloudLoader(filename=audio_file)
  audio = loader()
  spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
  w = Windowing(type = 'hann')
  fft = FFT() # this gives us a complex FFT

  pool = essentia.Pool()
  for frame in ess.FrameGenerator(audio, frameSize = 2048, hopSize = 2048, startFromZero=True):
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=13)(mag)
        contrast, spectralValley = SpectralContrast()(mag)
        flatness = Flatness()(mag)
        #dens = Welch()(spectrum(w(frame)))
        #onset = OnsetDetection()(mag,phase)
        #dynamic_complexity, loudness = DynamicComplexity()(mag)
        spectral_complex = SpectralComplexity()(mag)
        centroid = Centroid()(mag)
        #croma = Chromagram(sampleRate=2048*5)(mag[1:],)
        loudness = Loudness()(mag)

        pool.add('lowlevel.flatness', [flatness])
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', loudness)
        pool.add('lowlevel.spectralContrast', contrast)
        #pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.dyncomplex', [dynamic_complexity])
        pool.add('lowlevel.spectralComplexity', spectral_complex)
        #pool.add('lowlevel.chroma', croma)
        #pool.add('lowlevel.dens', dens)
        pool.add('lowlevel.centroid', centroid)

  # compute mean and variance of the frames
  aggrPool = PoolAggregator(defaultStats = ['mean', 'var'])(pool)

  # and ouput those results in a file
  YamlOutput(filename = 'features.json', format='json', writeVersion=False)(aggrPool)

 #['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'],
  #save_descriptors_as_strings():
  f_in = 'features.json'
  features = get_json(f_in)['lowlevel']['flatness']['mean']
  features.append(get_json(f_in)['lowlevel']['mfcc']['var'])
  features.append(get_json(f_in)['lowlevel']['spectralComplexity']['mean'])
  features.append(get_json(f_in)['lowlevel']['mfcc']['mean'])
  features.append(get_json(f_in)['lowlevel']['loudness']['mean'])
  features.append(get_json(f_in)['lowlevel']['centroid']['mean'])
  features.append(get_json(f_in)['lowlevel']['spectralContrast']['mean'])
 #features.append(get_json(f_in)['lowlevel']['onsets']['mean'])
 #features.append('0') //for testset
  features.append(class_number)
  f_out = csv_file
  f_out = open(f_out, 'a')
  writer = csv.writer(f_out)
  writer.writerow(features)

for root, dirs, files in os.walk(in_dir):
    dirs.sort()
    files.sort()
    path = root.split(os.sep)
    class_number = root.split('/')[-1] # to obtain the number of class
    print(class_number)
    root = os.path.basename(root)+'/'
    print(root)
    for file in files:
        print("\t", "Analyzing:", file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() == ".wav":
            audio_file = in_dir + root + file
            extract_features(audio_file)

print('Done')
