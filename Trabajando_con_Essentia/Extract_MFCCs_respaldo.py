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
file_out = open('anotatedMFCCsAsStrings.csv', 'w') #for erasing the file if already has data
#f_out = 'anotatedMFCCsAsStrings.csv'

def extract_features (path):
    loader = essentia.standard.MonoLoader(filename=path)
    audio = loader()
    mfcc = MFCC(numberCoefficients=13)
    loudness = Loudness()
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    w = Windowing(type = 'hann')

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        average_loudness = loudness(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', average_loudness)
        #pool.add('lowlevel.mfcc_bands', mfcc_bands)
        #pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))
    
    #YamlOutput(filename = 'mfcc.sig', format='yaml', writeVersion=False)(pool)

    # compute mean and variance of the frames
    #aggrPool = PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)
    aggrPool = PoolAggregator(defaultStats = ['mean'])(pool)

    # and ouput those results in a file
    YamlOutput(filename = 'features.json', format='json', writeVersion=False)(aggrPool)
    save_descriptors_as_strings()
    #class_number = get_annotated_data()
    #save_descriptors_as_strings(class_number)
    #get_annotated_data()

def open_files_and_get_class_numbers_from_dir(extract_features):
    for root, dirs, files in os.walk(in_dir):
        path = root.split(os.sep)
        #print((len(path) - 1) * '---', os.path.basename(root))
        #print((os.path.basename(root)+'/'))
        class_number = root.split('_')[-1] # to obtain the number of class
        #print(class_number)
        root = os.path.basename(root)+'/'
        for file in files:
            #print(len(path) * '---', file)
            file_name, file_extension = os.path.splitext(file)
            if file_extension.lower() == ".wav":
                #print(len(path) * '---', file)
                newpath = in_dir + root + file
    return (newpath, class_number)

def save_descriptors_as_strings():
    f_in = 'features.json'
    features = get_json(f_in)['lowlevel']['mfcc']['mean'] #['loudness']['mean']
    class_number = str(open_files_and_get_class_numbers_from_dir)[1]
    features.append(class_number)
    f_out = 'anotatedMFCCsAsStrings.csv'
    f_out = open(f_out, 'a')
    #with f_out:
    writer = csv.writer(f_out)
    writer.writerow(features)

#extract_features(open_files_and_get_class_numbers_from_dir()[0])
#open_files_and_get_class_numbers_from_dir(extract_features)[0]

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
file_out = open('anotatedMFCCsAsStrings.csv', 'w') #for erasing the file if already has data
#f_out = 'anotatedMFCCsAsStrings.csv'

def extract_features_2 (path):
    loader = essentia.standard.MonoLoader(filename=path)
    audio = loader()
    mfcc = MFCC(numberCoefficients=13)
    loudness = Loudness()
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    w = Windowing(type = 'hann')

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        average_loudness = loudness(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.loudness', average_loudness)
        #pool.add('lowlevel.mfcc_bands', mfcc_bands)
        #pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))
    
    #YamlOutput(filename = 'mfcc.sig', format='yaml', writeVersion=False)(pool)

    # compute mean and variance of the frames
    #aggrPool = PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)
    aggrPool = PoolAggregator(defaultStats = ['mean'])(pool)

    # and ouput those results in a file
    YamlOutput(filename = 'features.json', format='json', writeVersion=False)(aggrPool)

    #open files and get classes
    for root, dirs, files in os.walk(in_dir):
        path = root.split(os.sep)
        #print((len(path) - 1) * '---', os.path.basename(root))
        #print((os.path.basename(root)+'/'))
        class_number = root.split('_')[-1] # to obtain the number of class
        #print(class_number)
        root = os.path.basename(root)+'/'
        for file in files:
            #print(len(path) * '---', file)
            file_name, file_extension = os.path.splitext(file)
            if file_extension.lower() == ".wav":
                #print(len(path) * '---', file)
                newpath = in_dir + root + file

    #save_descriptors_as_strings():
    f_in = 'features.json'
    features = get_json(f_in)['lowlevel']['mfcc']['mean'] #['loudness']['mean']
    features.append(class_number)
    f_out = 'anotatedMFCCsAsStrings.csv'
    f_out = open(f_out, 'a')
    #with f_out:
    writer = csv.writer(f_out)
    writer.writerow(features)
    return(newpath)

path = extract_features_2()
extract_features_2(path)



def save_descriptors_as_matrix():
    file_name = 'mfccs.csv'
    f = open(file_name, 'w')
    file = open('mfccmean.sig').read()
    mfccs = file[35:]
    m = np.matrix(mfccs)
    print(mfccs)
    savetxt(f, m)

def features():
    for audio_files in sorted(glob.glob( 'audioClases/Clase_5/' + "*.wav" )):
        print(audio_files)
        mfccs = extract_mfccs(audio_files)

#features()


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