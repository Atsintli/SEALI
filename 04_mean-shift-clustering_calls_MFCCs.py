
import MFCC_extractor as xmfccs
from utils import save_descriptors_as_matrix 
import glob
# import librosa
import csv
import numpy as np
from numpy import savetxt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from numpy import loadtxt
import os
import matplotlib.pyplot as plt
from itertools import cycle
import shutil

# features is [[float]]
def meanShift(features, random_state=3):
    X, _ = make_blobs(n_samples=len(features),
                      n_features=24,
                      centers=[(8,8),(5,5),(3,3),(2,2),(1,1)],
                      #centers = [[1,1,1],[5,5,5],[3,10,10]],
                      #centers = features,
                      cluster_std=1.3,
                      shuffle=True,
                      random_state=random_state
                      )

    #X = features
    bandwidth = estimate_bandwidth(X, quantile=0.07,
                                   #n_samples=100,
                                   #random_state=4
                                   n_jobs=-1
    )

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False,
                   max_iter=10, cluster_all=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    return n_clusters_, labels, cluster_centers, X

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
            num = 'Segments/{:05d}'.format(elements)
            for audio_files in glob.glob(num + "*.wav"):
                shutil.copy(audio_files, 'audioClases/Clase_' + str(clase))
                print('moviendo archivo', audio_files, 'a',
                      'audioClasses/Clase_' + str(clase))

# features = list(map(lambda x: x["mean"], xmfccs.extract_all_mfccs(
#   glob.glob('seg/' + "*.wav"))))

# print (features)
# save_descriptors_as_matrix('dataBaseAsMatrix.csv', features)

features = loadtxt("dataBaseAsMatrix.csv")

a, b, c, d = meanShift(features)
# print(features, a, b)
# #writeFiles(n_clusters_=a, labels=b)
ploter(n_clusters_=a, labels=b, cluster_centers=c, X=d)
moveToFolders(a, b)