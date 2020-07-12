import json
import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv
<<<<<<< Updated upstream
from utils import get_json, save_as_json
=======
import os
>>>>>>> Stashed changes


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
<<<<<<< Updated upstream
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
=======
        #pool.add('lowlevel.mfcc_bands', mfcc_bands)
        #pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))
    
    #YamlOutput(filename = 'mfcc.sig', format='yaml', writeVersion=False)(pool)

    # compute mean and variance of the frames
    #aggrPool = PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)
    aggrPool = PoolAggregator(defaultStats = ['mean'])(pool)

    # and ouput those results in a file
    YamlOutput(filename = 'mfccmean.sig', format='yaml', writeVersion=False)(aggrPool)

    file = open('mfccmean.sig').read()
    mfccs = file[35:]
    m = np.matrix(mfccs)
    print(mfccs)
    savetxt(f, m)

file_name = 'mfccs.csv'
f = open(file_name, 'w')

def features():
    for audio_files in sorted(glob.glob( 'audioClases/Clase_5/' + "*.wav" )):
        print(audio_files)
        mfccs = extract_mfccs(audio_files)

features()

#estudiar csv!!!

def append_classes():
    f_in=open('mfccs.csv', 'r')
    f_out = 'mfccs_classes.csv'
    file = open(file_name, 'a')
    for line in f_in.readlines():
        print (line)
        #f_out.write(line.split(",")[0]+"")
        csv.writer(f_out, w, line)
    #file = open(file_name, 'w')

    # i=0
    # n_files=7 #number of files per folder
    # class_name=0 #class number extracted from folder
    # list = [] #list for apending features with class names
    # list = map(str, features())
    # while i < n_files: 
    #     for c in range(n_files): #class_name('audio'):
    #         print ('class_name:', c)
    #     file.write(', '.join(list) + ', ' + str(class_name) + '\n')
    #     file.close()

#append_classes()


def class_extractor(path):
    for root, dirs, files in os.walk(path):
        print('Number of classes:', len(dirs))
        print(dirs[0:len(dirs)])
        count_files = len(files)
        print (dirs)
        return(dirs, count_files)

#class_extractor('audioClases')

def mfccs_class():
    file_name = 'features2.csv'
    file = open(file_name, 'w')

    i=0
    n_files=7
    class_name=0
    list = []
    list = map(str, features())
    file = open(file_name, 'a')
    while i < n_files: 
        for c in range(n_files): #class_name('audio'):
            print ('class_name:', c)
        file.write(', '.join(list) + ', ' + str(class_name) + '\n') # el cero sería la clase
        file.close()

#mfccs_class()
>>>>>>> Stashed changes
