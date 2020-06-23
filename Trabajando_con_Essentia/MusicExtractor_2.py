import essentia
import essentia.standard as es
import glob
import csv
import numpy as np
import os

#rewriting the same file
file_name = 'features2.csv'
file = open(file_name, 'w')

# def get_className(path):
#     with os.scandir(path) as entries:
#         for entry in entries:
#             print(entry.name)
#         return(entry.name)

# def get_fileName_class_name(path):
#     for dirpath, dirnames, files in os.walk(path):
#         file_count = len(files)
#         print(file_count)
#         # print('Found directory:', dirpath)        
#         # if len(dirnames) > 1:
#         #     print('Number of classes:', len(dirnames))       
#         # print('Number of files:', file_count)
#         print(dirnames)
#         for file_name in files:
#              print(file_name)
#         #     for classname in (dirnames):
#         #         print('Class Name:', classname) 
#         # print('\n')
#         return(file_count, dirnames)

list = []

'''
for dirpath, dirnames, files in os.walk('audio'):
    dirnames = [n for n in dirnames]
    contents = dirnames + files
    for f in contents:
        file_count = len(files)
        list.append((dirnames, file_count))
print(list)
'''

'''
for dirpath, dirnames, files in os.walk('audio'):
    #files = [n for n in files]
    file_count = [len(files)]
    contents = dirnames + file_count
    list.append((contents))
    print (list)
'''

'''
for dirpath, dirnames, files in os.walk('audio'):
    files = [n for n in files]
    file_count = len(files)
    #contents = dirnames + file_count
    list.append((dirnames, file_count))
    print(list)
'''
for dirpath, dirnames, files in os.walk('audio'):
    #files = [n for n in files]
    #file_count = [len(files)]
    #contents = dirnames + file_count
    #list.append((contents))
    for name in dirnames:
        print(dirnames)
    for file in files:
        print(files)


#get_fileName_class_name('audio')
#print(get_fileName_class_name('audio/'))

# def loop_data(path):
#     path, dirs, files = next(os.walk(path))
#     file_count = len(files)
#     for i in class_name():
#         print(file_count)

# for dirpath, dirnames, files in os.walk('audio/', topdown=False):
#     print('Found directory:', dirpath)
#     for file_name in files:
#         print(file_name)


# # Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
# def feature_extractor(audio_files):
#     f, features_frames = es.MusicExtractor(
#         lowlevelStats=['mean', 'stdev'],
#         rhythmStats=['mean', 'stdev'],
#         tonalStats=['mean', 'stdev'])(audio_files)

#     print("Filename:", f['metadata.tags.file_name'])
#     #print(sorted(f.descriptorNames()))

#     loudness = f['lowlevel.average_loudness']
#     barkbands = f['lowlevel.barkbands.mean']
#     tuning = f['tonal.tuning_frequency']
#     flatness = f['lowlevel.barkbands_flatness_db.mean']
#     onset = f['rhythm.onset_rate']
#     rhythm = f['rhythm.bpm_histogram']
#     #beats_count = f['rhythm.beats_count'] # la escala esta por encima de 1 habŕia que normalizar los valores
#     mel = f['lowlevel.melbands_crest.mean']
#     mfcc = f['lowlevel.mfcc.mean']
#     entropy = f['lowlevel.spectral_entropy.mean']

#     features = np.hstack([loudness,barkbands,tuning,flatness,onset,mel,mfcc,entropy, rhythm])
#     n_descriptors = len(features)
#     print('Number of descriptors: ' + str(n_descriptors))

#     #class_name('audio')

#     list = []
#     list = map(str, features)
#     file = open(file_name, 'a')
#     while i < n_files: 
#         for c in class_name('audio'):
#             print ('class_name:', c)
#         file.write(', '.join(list) + ', i' + '\n') # el cero sería la clase
#         file.close()

# #def get_dataset():
# for audio_files in glob.glob('audio/1' + "/" + "*.wav" ):
#     feature_extractor(audio_files)

# #feature_extractor('audio/dumitrescu.wav')

# # for audio_files in glob.glob( 'audio' + "/" + "*.wav" ):
# #     feature_extractor(audio_files)

