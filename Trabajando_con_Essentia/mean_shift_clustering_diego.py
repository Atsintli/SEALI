import Extract_MFCCs as xmfccs
import glob
# import librosa
import csv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from numpy import loadtxt
import os
import matplotlib.pyplot as plt
from itertools import cycle
import shutil


# features is [[float]]
def meanShift(features, cluster_std, random_state=None):
    X, _ = make_blobs(n_samples=len(features),
                      n_features=13,
                      centers=features,
                      cluster_std=cluster_std,
                      #   shuffle=False,
                      random_state=random_state
                      )

    bandwidth = estimate_bandwidth(X, quantile=0.5,
                                   n_samples=6)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True,
                   max_iter=5000, cluster_all=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)
    return n_clusters_, labels, cluster_centers, X


# features = list(map(lambda x: x["mean"], xmfccs.extract_all_mfccs(
#     glob.glob('/home/diego/sc/overtone/taller-abierto/resources/samples/humedad_8_melodic_split_2020_06_30/' + "*.wav"))))
# # features = loadtxt("dataBaseAsMatrix.csv")

# a, b, c, d = meanShift(features)
# print(features, a, b)
# #writeFiles(n_clusters_=a, labels=b)
# #ploter(n_clusters_=a, labels=b, cluster_centers=c, X=d)
# # moveToFolders(a, b)
