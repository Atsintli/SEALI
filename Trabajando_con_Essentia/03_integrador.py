import sys
import os
import json
import csv
import numpy as np
from numpy import savetxt

in_dir = 'audio_descriptors_extended'
print(in_dir)
todoDatos = []

def integrator(root, file):
	file_name, file_extension = os.path.splitext(file)
	with open(root + '/' + file) as f:
		data = json.load(f)
		datosDeUno = []
		#datosDeUno.append(file_name)
		datosDeUno.append(data.get('lowlevel').get('average_loudness'))
		datosDeUno.append(data.get('lowlevel').get('mfcc').get('mean'))
		#datosDeUno.append(0) #This is for the class in string mode
		#aqui poner mas descriptores
		datosDeUnoFlatten = [a for x in datosDeUno for a in (x if isinstance(x, list) else [x])]
		todoDatos.append(datosDeUnoFlatten)

def save_descriptors_as_matrix(file_name):
	f = open(file_name, 'w')
	m = np.matrix(todoDatos)
	savetxt(f, m)
	print('Done')

def save_descriptors_as_strings(file_name):
	file_name  = open(file_name, 'w')
	with file_name:
		writer = csv.writer(file_name)
		for row in todoDatos:
			writer.writerow(row)

for root, dirs, files in os.walk(in_dir):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        #print(len(path) * '---', file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() == ".json":
        	print(file)
        	integrator(root,file)

save_descriptors_as_matrix("dataBaseAsMatrix.csv")
save_descriptors_as_strings("dataBaseAsStrings.csv")