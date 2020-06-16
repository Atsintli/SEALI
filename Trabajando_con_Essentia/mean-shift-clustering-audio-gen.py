print(__doc__)

import librosa
import csv
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from numpy import loadtxt
import os

# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

points = loadtxt('features.csv')
print (len(points))
print()
X, _ = make_blobs(n_samples=len(points), n_features=12, centers=points, cluster_std=0.33, shuffle=True, random_state=1546481)


# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=20)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=1000,cluster_all=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

#############################################################################33
#Write files

files = []

for root_dir_path, sub_dirs, file in os.walk('Segments'):
    for f in file: # files need to be converted to strings for join
        file = f.split('_')[0]
        files.append((file))
#print(type(files))

with open('archivos_clases.txt', 'w') as f:
  writer = csv.writer(f, delimiter=' ')
  writer.writerows(zip(files,labels))

#concatenate classes in archives

audioname = 'test'
clases = open('archivos_clases.txt')
clasescontent = clases.readlines()

clases = [int(x.split(" ")[1]) for x in clasescontent]
#print (clases)

for clase in range(n_clusters_):
	print("iterando sobre " + str(clase))
	ele = np.where(np.array(clases)==clase)[0]
	print("indices de clase " + str(clase) + " son ")
	#print(ele)
	audiototal = np.array([])
	for elements in ele:
		conStr = '{:04d}'.format(elements)
    #nomArchivo = audioname + "/" + audioname + "_" + conStr + ".wav"
		nomArchivo = "Segments/" + conStr + "_segment" + ".wav"
		print("leyendo " + nomArchivo)
		y, sr = librosa.load(nomArchivo)
		audiototal = np.append(audiototal,y)
	librosa.output.write_wav("Clusters/" + "CLASE_" 
		+ str(clase) + ".wav", audiototal,sr)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()