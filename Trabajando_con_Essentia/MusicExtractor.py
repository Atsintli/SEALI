# -*- coding: utf-8 -*- 

import essentia
import essentia.standard as es
import glob
import csv
import numpy as np
import os
from numpy import savetxt

#rewriting the same file
file_name = 'features.csv'
file = open(file_name, 'w')

# Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
def feature_extractor(audio_files):
    f, features_frames = es.MusicExtractor(
        lowlevelStats=['mean'],
        rhythmStats=['mean'],
        tonalStats=['mean'])(audio_files)

    print("Filename:", f['metadata.tags.file_name'])
    #print(sorted(f.descriptorNames()))

    #loudness = f['lowlevel.average_loudness']
    barkbands = f['lowlevel.barkbands.mean']
    tuning = f['tonal.tuning_frequency']
    flatness = f['lowlevel.barkbands_flatness_db.mean']
    #onset = f['rhythm.onset_rate']
    rhythm = f['rhythm.bpm_histogram']
    #beats_count = f['rhythm.beats_count'] # la escala esta por encima de 1 habŕia que normalizar los valores
    mel = f['lowlevel.melbands_crest.mean']
    mfcc = f['lowlevel.mfcc.mean']
    entropy = f['lowlevel.spectral_entropy.mean']

    print(mfcc)
    features = np.hstack([mfcc])
    #features = np.hstack([loudness])
    n_descriptors = len(features)
    print('Number of descriptors: ' + str(n_descriptors))

    b = np.matrix(features)
    savetxt(file, b)

def class_extractor(path):
    for root, dirs, files in os.walk(path):
        print('Number of classes:', len(dirs))
        print(dirs[0:len(dirs)])
        clases = 0
        count_files = len(files)
        print (dirs)
        return(dirs, count_files)

class_extractor('audioClases')

# #feature_extractor('audio/0/dumitrescu.wav')

# for audio_files in glob.glob( 'audio/' + "0/" + "*.wav" ):
#      feature_extractor(audio_files)

# for audio_files in glob.glob( 'audio/' + "*.wav" ):
#      feature_extractor(audio_files)