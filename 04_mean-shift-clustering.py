import Extract_MFCCs as xmfccs
import glob
import librosa
import csv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from numpy import loadtxt
import os
import matplotlib.pyplot as plt
from itertools import cycle
import shutil

def meanShift(features):  # features is [[float]]
    # X, _ = make_blobs(n_samples=len(features),
    #                     n_features=13,
    #                     centers=features, 
    #                     cluster_std=0.1,
    #                     shuffle=True,
    #                     random_state=True
    #                     )

    bandwidth = estimate_bandwidth(features, quantile=0.1, 
                                    n_samples=900)

    ms = MeanShift(
                bandwidth=bandwidth, 
                bin_seeding=True,
                max_iter=5000,
                cluster_all=True)
    ms.fit(features,)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    #print("number of estimated clusters : %d" % n_clusters_)
    #P = meanshift.predict(features)

    return n_clusters_, labels, cluster_centers, features

###Write files by the number of clusters###

def writeFiles(n_clusters_, labels):
    folder = 'Clusters/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    files = []

    for audio_files in sorted(glob.glob( 'Segments/' + "*.wav" )):
        names = audio_files.split('/')[1]
        #print(names)
        files.append(names)

    with open('archivos_clases.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(files, labels))

    # concatenate classes in archives
    clases = open('archivos_clases.txt')
    clasescontent = clases.readlines()
    clases = [int(x.split(" ")[1]) for x in clasescontent]

    for clase in range(n_clusters_):
        print("iterando sobre " + str(clase))
        ele = np.where(np.array(clases) == clase)[0]
        print("indices de clase " + str(clase) + " son: " + str(ele))
        #print(ele)
        audiototal = np.array([])
        for elements in ele:
            num = 'Segments/{:06d}'.format(elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                #print(audiototal)
                librosa.output.write_wav("Clusters/" + "CLASE_"
                                         + str(clase) + ".wav", audiototal, sr)
            #print(audiototal)


def moveToFolders(n_clusters_, labels):
    files = []
    for audio_files in sorted(glob.glob( 'Segments/' + "*.wav" )):
        names = audio_files.split('/')[1]
        #print(names)
        files.append(names)

    with open('archivos_clases.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(files, labels))

    clasescontent = open('archivos_clases.txt').readlines()
    clases = [int(x.split(" ")[1]) for x in clasescontent]

    for createFolders in range(n_clusters_):
        folder = 'audioClases/Clase_' + str(createFolders)
        if not os.path.exists(folder):
            os.makedirs(folder)

    for clase in range(n_clusters_):
        ele = np.where(np.array(clases) == clase)[0]
        print("indices de clase " + str(clase) + " son: " + str(ele))
        for elements in ele:
            num = 'Segments/{:06d}'.format(elements)
            for audio_files in glob.glob(num + "*.wav"):
                shutil.copy(audio_files, 'audioClases/Clase_' + str(clase))
                print('moviendo archivo', audio_files, 'a',
                      'audioClases/Clase_' + str(clase))

# Plot result

def ploter(n_clusters_, labels, cluster_centers, X):
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


features = list(xmfccs.extract_all_mfccs(glob.glob('Segments/' + "*.wav")))

features = loadtxt("dataBaseAsMatrix.csv")

a, b, c, d = meanShift(features)
#writeFiles(n_clusters_=a, labels=b)
ploter(n_clusters_=a, labels=b, cluster_centers=c, X=d)
#moveToFolders(a, b)
