#%%
import glob
#import librosa
import csv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from numpy import loadtxt
from numpy import savetxt
import os
import matplotlib.pyplot as plt
from itertools import cycle
import shutil
from csv import writer
from csv import reader

def meanShift(features):  #features is [[float]]
    #Standardize the feature matrix
    #tr_features = StandardScaler().fit_transform(features) #quite .data de features
    #Create a PCA
    pca = PCA(n_components=2,)# whiten=True) # svd_solver="randomized"

    features_pca = pca.fit_transform(features)

    print(features.shape[1])
    print(len(features_pca))
    print (features_pca.shape[1])
    print(features_pca[0:5])

    bandwidth = estimate_bandwidth(features_pca, quantile=0.04,
                                    n_samples=None)

    ms = MeanShift(
                bandwidth=bandwidth,
                bin_seeding=True,
                max_iter=1000,
                cluster_all=True)
    ms.fit(features_pca)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    #print("number of estimated clusters : %d" % n_clusters_)
    P = ms.predict(features_pca)
    return n_clusters_, labels, cluster_centers, features_pca

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
            num = folder_in+'{:06d}'.format(elements)
            for audio_files in glob.glob(num + "*.wav"):
                shutil.copy(audio_files, folder_out + str(clase))
                print('moviendo archivo', audio_files, 'a',
                      folder_out + str(clase))

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

def assing_label_to_dataset(input_file, output_file, transform_row):
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)


file_in = "sink_into_return.csv"
folder_in = "segments_music18/"
folder_out = "clusters_music18/"
features = loadtxt(file_in)
a, b, c, d = meanShift(features)

#%%
#writeFiles(n_clusters_=a, labels=b)
#savetxt("centros.csv",c)
#moveToFolders(a, b, folder_in=folder_in, folder_out=folder_out)
ploter(n_clusters_=a, labels=b, cluster_centers=c, X=d)
#csv_fileOut = "dataset_label_test.csv"
#labels = b.tolist()
#print("soy len", len(labels))
#assing_label_to_dataset(file_in, csv_fileOut, lambda row, line_num: row.append(b[line_num -1]))

# %%
