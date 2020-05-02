import essentia
import essentia.standard as es
import glob
import csv
import numpy as np
import os

#rewriting the same file
file_name = 'features2.csv'
file = open(file_name, 'w')

# Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
def feature_extractor(audio_files):
    f, features_frames = es.MusicExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean', 'stdev'],
        tonalStats=['mean', 'stdev'])(audio_files)

    print("Filename:", f['metadata.tags.file_name'])
    #print(sorted(f.descriptorNames()))

    loudness = f['lowlevel.average_loudness']
    barkbands = f['lowlevel.barkbands.mean']
    tuning = f['tonal.tuning_frequency']
    flatness = f['lowlevel.barkbands_flatness_db.mean']
    onset = f['rhythm.onset_rate']
    rhythm = f['rhythm.bpm_histogram']
    #beats_count = f['rhythm.beats_count'] # la escala esta por encima de 1 habŕia que normalizar los valores
    mel = f['lowlevel.melbands_crest.mean']
    mfcc = f['lowlevel.mfcc.mean']
    entropy = f['lowlevel.spectral_entropy.mean']

    features = np.hstack([loudness,barkbands,tuning,flatness,onset,mel,mfcc,entropy, rhythm])
    n_descriptors = len(features)
    print('Number of descriptors: ' + str(n_descriptors))

    list = []
    list = map(str, features)
    file = open(file_name, 'a')
    file.write(', '.join(list) + ', 0' + '\n') # el cero sería la clase
    file.close()

def class_extractor(path):
    for root, dirs, files in os.walk(path):
        #print('Number of classes:', len(dirs))
        #print(dirs[0:1], dirs[1:2], dirs[2:3], dirs[3:4])
        count_files = len(files)
        return(dirs, count_files)

print(class_extractor('audio'))

#feature_extractor('audio/dumitrescu.wav')

# for audio_files in glob.glob( 'audio' + "/" + "*.wav" ):
#     feature_extractor(audio_files)