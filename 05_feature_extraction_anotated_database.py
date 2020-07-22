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

in_dir = 'audioClases/'
file_out = open('anotatedDataBase.csv', 'w') #for erasing the file if already has data
#f_out = 'anotatedMFCCsAsStrings.csv'

def extract_features(path):
  loader = ess.MonoLoader(filename=path)
  audio = loader()
  mfcc = MFCC(numberCoefficients=13)
  #loudness = Loudness()
  spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
  w = Windowing(type = 'hann')

  pool = essentia.Pool()
  for frame in ess.FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
      mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
      #average_loudness = loudness(spectrum(w(frame)))
      pool.add('lowlevel.mfcc', mfcc_coeffs)
      #pool.add('lowlevel.loudness', average_loudness)
      #pool.add('lowlevel.mfcc_bands', mfcc_bands)
      #pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))

  #YamlOutput(filename = 'mfcc.sig', format='yaml', writeVersion=False)(pool)

  # compute mean and variance of the frames
  #aggrPool = PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)
  aggrPool = PoolAggregator(defaultStats = ['mean'])(pool)

  # and ouput those results in a file
  YamlOutput(filename = 'features.json', format='json', writeVersion=False)(aggrPool)

  #save_descriptors_as_strings():
  f_in = 'features.json'
  features = get_json(f_in)['lowlevel']['mfcc']['mean']#['loudness']['mean']
  features.append(class_number)
  f_out = 'anotatedDataBase.csv'
  f_out = open(f_out, 'a')
  writer = csv.writer(f_out)
  writer.writerow(features)

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

#paths = []
#pendiente . . . 
# def open_files_and_get_classes():
#     #paths = []
#     for root, dirs, files in os.walk(in_dir):
#         path = root.split(os.sep)
#         #print((len(path) - 1) * '---', os.path.basename(root))
#         #print((os.path.basename(root)+'/'))
#         class_number = root.split('_')[-1] # to obtain the number of class
#         #print(class_number)
#         root = os.path.basename(root)+'/'
#         for file in files:
#             #print(len(path) * '---', file)
#             file_name, file_extension = os.path.splitext(file)
#             if file_extension.lower() == ".wav":
#                 #print(len(path) * '---', file)
#                 newpath = in_dir + root + file
#                 #extract_features(newpath)
#                 paths.append(newpath)
#     return paths

#print(open_files_and_get_classes())
#open_files_and_get_classes()
#list(map(extract_features, paths))

