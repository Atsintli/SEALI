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

def meanShift():
	points = loadtxt('mfccs.csv')
	X, _ = make_blobs(n_samples=len(points), n_features=13, centers=points, cluster_std=1.0)
	# Compute clustering with MeanShift
	# The following bandwidth can be automatically detected using

	bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=2000)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=1000,cluster_all=True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)

	print("number of estimated clusters : %d" % n_clusters_)
	return n_clusters_, labels, cluster_centers, X

###Write files into n clusters archives###
def writeFiles(n_clusters_, labels):
	folder = 'Clusters/'
	if not os.path.exists(folder):
		os.makedirs(folder)
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
	clases = open('archivos_clases.txt')
	clasescontent = clases.readlines()
	clases = [int(x.split(" ")[1]) for x in clasescontent]

	for clase in range(n_clusters_):
		print("iterando sobre " + str(clase))
		ele = np.where(np.array(clases)==clase)[0]
		print("indices de clase " + str(clase) + " son: " + str(ele))
		#print(ele)
		audiototal = np.array([])
		for elements in ele:
			num = 'Segments/{:04d}'.format(elements)
			for audio_files in glob.glob(num + "*.wav" ):
				print("leyendo " + audio_files)
				y, sr = librosa.load(audio_files)
				audiototal = np.append(audiototal, y)
				#print(audiototal)
				librosa.output.write_wav("Clusters/" + "CLASE_" 
				+ str(clase) + ".wav", audiototal, sr)
			#print(audiototal)

def moveToFolders(n_clusters_, labels):

	files = []
	for root_dir_path, sub_dirs, file in os.walk('Segments'):
		for f in file: # files need to be converted to strings for join
				file = f.split('_')[0]
				files.append((file))

	with open('archivos_clases.txt', 'w') as f:
		writer = csv.writer(f, delimiter=' ')
		writer.writerows(zip(files,labels))

	clasescontent = open('archivos_clases.txt').readlines()
	clases = [int(x.split(" ")[1]) for x in clasescontent]

	for createFolders in range(n_clusters_):
		folder = 'audioClases/Clase_' + str(createFolders)
		if not os.path.exists(folder):
			os.makedirs(folder)

	for clase in range(n_clusters_):
		ele = np.where(np.array(clases)==clase)[0]
		print("indices de clase " + str(clase) + " son: " + str(ele))
		for elements in ele:
			num = 'Segments/{:04d}'.format(elements)
			for audio_files in glob.glob( num + "*.wav" ):
				shutil.copy(audio_files, 'audioClases/Clase_' + str(clase))
				print('moviendo archivo', audio_files, 'a', 'audioClases/Clase_' + str(clase))

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

a,b,c,d = meanShift()
#writeFiles(n_clusters_=a, labels=b)
ploter(n_clusters_=a,labels=b,cluster_centers=c,X=d)
moveToFolders(a,b)