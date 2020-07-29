import sys
import os
import json
import csv
import numpy as np
from numpy import savetxt

in_dir = 'audio_descriptors_extended'
all_data = []

def integrator(root, file):
	file_name, file_extension = os.path.splitext(file)
	with open(root + '/' + file) as f:
		data = json.load(f)
		datos = []
		#datos.append(file_name)
		datos.append(data.get('lowlevel').get('mfcc').get('mean'))
		datos.append(data.get('lowlevel').get('average_loudness'))
		#datos.append(0) #This is for the class in string mode
		#aqui poner mas descriptores
		datosFlatten = [a for x in datos for a in (x if isinstance(x, list) else [x])]
		all_data.append(datosFlatten)

def save_descriptors_as_strings(file_name):
	file_name  = open(file_name, 'w')
	with file_name:
		writer = csv.writer(file_name)
		for row in all_data:
			writer.writerow(row)

def iter_files():
	for root, dirs, files in os.walk(in_dir):
			path = root.split(os.sep)
			print((len(path) - 1) * '---', os.path.basename(root))
			for file in files:
					#print(len(path) * '---', file)
					file_name, file_extension = os.path.splitext(file)
					if file_extension.lower() == ".json":
						print(file)
						integrator(root,file)

def save_matrix_array(file_name, matrix):
    with open(file_name, 'w') as f:
      savetxt(f, matrix)
    print('Done')

iter_files()
save_matrix_array("dataBaseAsMatrix_3.csv", all_data)


