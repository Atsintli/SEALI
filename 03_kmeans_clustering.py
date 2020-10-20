import shutil
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
from numpy import loadtxt
import os
from numpy import savetxt
import csv
import librosa
import glob

# load array

folder = "audioClases_kMeans/"
try:
    shutil.rmtree(folder)
except OSError as e:
    print ("Error: %s - %s." % (e.filename, e.strerror))

#points = loadtxt('dataBaseAsMatrix_2.csv')
points = loadtxt('dataBaseAsMatrix_standard_mel_test.csv')

def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=100)

len_X = (len(input_fn()))

num_clusters = 8
kmeans = tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=True, feature_columns=None)

# train
num_iterations = 50
previous_centers = None 
for _ in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print()
    #print ('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print ('score:', kmeans.score(input_fn))
#print ('cluster centers:', cluster_centers)

#map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(points):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  #print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)

def moveToFolders(n_clusters_, labels):
    files = []
    for audio_files in sorted(glob.glob( 'Segmentstest/' + "*.wav" )):
        names = audio_files.split('/')[1]
        #print(names)
        files.append(names)

    with open('archivos_clases.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(files, labels))

    clasescontent = open('archivos_clases.txt').readlines()
    clases = [int(x.split(" ")[1]) for x in clasescontent]

    for createFolders in range(n_clusters_):
        folder = 'audioClases_kMeans/Clase_' + str(createFolders)
        if not os.path.exists(folder):
            os.makedirs(folder)

    for clase in range(n_clusters_):
        ele = np.where(np.array(clases) == clase)[0]
        print("indices de clase " + str(clase) + " son: " + str(ele))
        for elements in ele:
            num = 'Segmentstest/{:05d}'.format(elements)
            for audio_files in glob.glob(num + "*.wav"):
                shutil.copy(audio_files, 'audioClases_kMeans/Clase_' + str(clase))
                print('moviendo archivo', audio_files, 'a',
                      'audioClases_kMeans/Clase_' + str(clase))

moveToFolders(num_clusters, cluster_indices)

