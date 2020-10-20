import sys
import os
from essentia.standard import *
import numpy as np
import json
import essentia
import essentia.standard as ess
import matplotlib.pyplot as plt
from numpy import savetxt
import glob
import csv
import os
from utils import *
import toolz as tz
import librosa
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from numpy import loadtxt
from itertools import cycle
import shutil
import soundfile
import time

begin_time = time.time()


counter = 0

def segments_gen(fileName):

    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    
    loader = essentia.standard.MonoLoader(filename=fileName)
    audio = loader()
    w = Windowing(type = 'hann')
    spectrum = Spectrum()
    mfcc = MFCC()

    logNorm = UnaryOperator(type='log')
    pool = essentia.Pool()
    print("num de samples", len(audio))
    for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)

    #pedazos muy pequenos
    minimumSegmentsLength = 10
    size1 = 300
    inc1 = 60
    size2 = 200
    inc2 = 60
    cpw = 9.5

    features = [val for val in pool['lowlevel.mfcc'].transpose()]

    sbic = SBic(size1=size1, inc1=inc1,size2=size2, inc2=inc2,cpw=cpw, minLength=minimumSegmentsLength)
    segments = sbic(np.array(features))
    record_segments(audio,segments)

def record_segments(audio, segments):
	for segment_index in range(len(segments) - 1):
		global counter
		start_position = int(segments[segment_index] * 512)
		end_position = int(segments[segment_index + 1] * 512)
		writer = essentia.standard.MonoWriter(filename=segments_dir + "{:06d}".format(counter) + ".wav", format="wav")(audio[start_position:end_position])
		counter = counter + 1

def read_files_gen_segments():
    for root, dirs, files in os.walk(rootDir):
        files.sort()
        path = root.split(os.sep)
        for file in files:
                file_name, file_extension = os.path.splitext(file)
                if file_extension.lower() == ".wav":
                    print('Cuting: ', file)
                    segments_gen(root + "/" + file)

#extract features

def extract_mfccs(audio_file):
    #loader = essentia.standard.MonoLoader(filename=audio_file)
    loader = essentia.standard.EqloudLoader(filename=audio_file)
    print("Extracting Features: " + audio_file)
    audio = loader()
    spectrum = Spectrum()
    melBands = MelBands()
    w = Windowing(type='hann')
    fft = FFT() # this gives us a complex FFT

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize=2048, hopSize=2048, startFromZero=True): #for chroma frameSize=8192*2, hopSize=8192, #fz=88200, hs=44100
        mag, phase, = CartesianToPolar()(fft(w(frame)))
        mfcc_bands, mfcc_coeffs = MFCC(numberCoefficients=number_of_mfccs)(mag)
        barkbands = BarkBands()(mag)
        #mel_bands = melBands(spectrum(w(frame)))
        # contrast, spectralValley = SpectralContrast()(mag)
        #flatness = Flatness()(mag) 
        onset = OnsetDetection()(mag,phase)
        #dynamic_complexity, loudness = DynamicComplexity()(mag)
        #spectral_complex = SpectralComplexity()(mag)
        #croma = Chromagram(sampleRate=2048*5)(mag[1:],)

        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.barkbands', barkbands)
        #pool.add('lowlevel.loudness', [loudness])
        #pool.add('lowlevel.melbands', mel_bands)
        #pool.add('lowlevel.spectralcontrast', contrast)
        #pool.add('lowlevel.flatness', [flatness])
        pool.add('lowlevel.onsets', [onset])
        #pool.add('lowlevel.dyncomplex', [dynamic_complexity])
        #pool.add('lowlevel.spectral_complexity', [spectral_complex])
        #pool.add('lowlevel.chroma', croma)

    aggrPool = PoolAggregator(defaultStats=['mean'])(pool)

    YamlOutput(filename='features.json', format='json',
               writeVersion=False)(aggrPool)

    json_data = get_json("features.json")
    #dyncomp = json_data['lowlevel']['dynamic_complexity']['mean']

    #[[MFCC],[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],[BeatStatistics]];

    #os.remove("mfccmean.json")
    return {"file": audio_file, 
            "mfccMean": json_data['lowlevel']['mfcc']['mean'],
            "barkbands": json_data['lowlevel']['barkbands']['mean'],
            #"mel": json_data['lowlevel']['melbands']['mean'], 
            # "loudness": json_data['lowlevel']['loudness']['mean'],
            # "spectralContrast": json_data['lowlevel']['spectralcontrast']['mean'],
            # "chroma": json_data['lowlevel']['chroma']['mean'],
            #"flatness": json_data['lowlevel']['flatness']['mean'],
            "onsets": json_data['lowlevel']['onsets']['mean'],
            #"dyncomplexity": json_data['lowlevel']['dyncomplex']['mean'],
            #"complexity": json_data['lowlevel']['spectral_complexity']['mean']
            }

def extract_all_mfccs(audio_files):
    return list(map(extract_mfccs, audio_files))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data: 
               list(tz.concat(getProps(
                   #['flatness', 'complexity', 'dyncomplexity','mfccMean','onsets'], 
                   #['mfccMean','flatness', 'complexity', 'onsets'],
                   #['mel'],
                   ['barkbands', 'onsets', 'mfccMean'],
                   data))),
    input_data))
    return features

def save_as_matrix(features):
    save_descriptors_as_matrix(features_csv, features)

# Mean shift Clustering

def meanShift(features, quantiles, samples, seeding):

    #queda pendiente que no sabemos si cuantos features está tomando realmente en cuenta
    bandwidth = estimate_bandwidth(features, quantile=quantiles, 
                                    n_samples=samples,
                                    n_jobs=-1)

    ms = MeanShift(
                bandwidth=bandwidth,
                seeds=features,
                bin_seeding=seeding,
                max_iter=300,
                cluster_all=True)
    ms.fit(features)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    #print("number of estimated clusters : %d" % n_clusters_)
    P = ms.predict(features)
    return n_clusters_, labels, cluster_centers, features

###Write files by the number of clusters###

def writeFiles(n_clusters_, labels, folder_out):
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    files = []

    for audio_files in sorted(glob.glob( segments_dir + "*.wav" )):
        names = audio_files.split('/')[1]
        files.append(names)

    with open('archivos_clases.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(files, labels))

    clases = open('archivos_clases.txt')
    clasescontent = clases.readlines()
    clases = [int(x.split(" ")[1]) for x in clasescontent]

    for clase in range(n_clusters_):
        print("iterando sobre " + str(clase))
        ele = np.where(np.array(clases) == clase)[0]
        print("indices de clase " + str(clase) + " son: " + str(ele))
        audiototal = np.array([])
        for elements in ele:
            num = segments_dir+'{:06d}'.format(elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                soundfile.write(folder_out + "CLASE_" + str(clase) + ".wav", audiototal, sr)
                #essentia.standard.MonoWriter(filename=folder_out + "CLASE_" + str(clase) + ".wav", format="wav")(audiototal)

def moveToFolders(n_clusters_, labels, folder_in, folder_out):
    files = []
    for audio_files in sorted(glob.glob(folder_in + "*.wav" )):
        names = audio_files.split('/')[1]
        #print(names)
        files.append(names)

    with open('archivos_clases.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(files, labels))

    clasescontent = open('archivos_clases.txt').readlines()
    clases = [int(x.split(" ")[1]) for x in clasescontent]

    for createFolders in range(n_clusters_):
        folder = folder_out + str(createFolders)
        if not os.path.exists(folder):
            os.makedirs(folder)

    for clase in range(n_clusters_):
        ele = np.where(np.array(clases) == clase)[0]
        print("indices de clase " + str(clase) + " son: " + str(ele))
        for elements in ele:
            num = folder_in+'{:04d}'.format(elements)
            for audio_files in glob.glob(num + "*.wav"):
                shutil.copy(audio_files, folder_out + str(clase))
                print('moviendo archivo', audio_files, 'a',
                      folder_out + str(clase))



# Plot result

def ploter(n_clusters_, labels, cluster_centers, data):
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

print('Done')

#Run
rootDir = "/media/atsintli/Samsung/audioClases/fijo/"
segments_dir = '/media/atsintli/Samsung/audioClases/fijo/'
#features_csv = 'dataBaseAsMatrix_scmir.csv'
#MSC_folder = '/media/atsintli/Samsung/audioClases/Clase_10a/Q_0.1_scmir/'

#number_of_mfccs = 12
read_files_gen_segments()

#input_data = extract_all_mfccs(sorted(glob.glob(segments_dir + "*.wav")))
#save_as_matrix(concat_features(input_data))

#features = loadtxt(features_csv)
#n_clusters_, labels, cluster_centers, features
#a, b, c, d = meanShift(features, quantiles=0.3, samples=len(features), seeding=False)
#ploter(n_clusters_=a, labels=b, cluster_centers=c, data=d)
#writeFiles(n_clusters_=a, labels=b, folder_out='GluedAudioClass/')
#moveToFolders(n_clusters_=a, labels=b, folder_in=segments_dir, folder_out=MSC_folder)

end_time = time.time()
print("Total time (s)", end_time- begin_time)