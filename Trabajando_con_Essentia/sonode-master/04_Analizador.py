import sys
import os
import json
import csv
import numpy as np
from sklearn.cluster import KMeans

#Python3 + Analizador.py + entrada en CSV + nombre salida + numero de clases

archivoDeEntrada = 'dataBaseAsStrings.csv'
archivoDeSalida = 'KMeans.wav'
numeroDeClusters = 2

print(archivoDeEntrada)
	
todoDatos = []
nombresArchivo = []

file = open(archivoDeEntrada, 'r')
lines = file.readlines() 

# Strips the newline character 
for line in lines:
	lineaCortada = line.strip().split(",")
	lineaFormateada = np.array(lineaCortada[1:]).astype(np.float)
	todoDatos.append(lineaFormateada)
	nombresArchivo.append(lineaCortada[0])

#normalizar
def normalize_columns(arr):
    rows, cols = arr.shape
    for col in range(cols):
        arr[:,col] /= abs(arr[:,col]).max()

todoDatos = np.array(todoDatos)
print(todoDatos)
normalize_columns(todoDatos)
print("*********************************")
print(todoDatos)

# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=numeroDeClusters, n_init=20, n_jobs=4)
# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(todoDatos)
# Evaluate the K-Means clustering accuracy.
print(y_pred_kmeans)

salida = open(archivoDeSalida, 'w')
nombresArchivo.sort()
for nombre in nombresArchivo:
	index = int(nombre)
	salida.write(nombre.zfill(6) + 1*" " + str(y_pred_kmeans[index]) + "\n")
print("done cambiado")