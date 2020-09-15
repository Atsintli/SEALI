import sys
import os
import subprocess, glob
import json
from numpy import savetxt
import glob

#in_dir = 'audioClases/'
in_dir = 'Segments_2/'
out_dir = 'audio_descriptors_extended/'
in_dir_2 = out_dir
file_out = "dataBaseAsMatrix_streaming.csv"
all_data = []

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

#Important note. This is not performing very well with the mean shift clustering algorithm
def extractor(file):
    path = os.path.basename(file)
    get_fileName = os.path.splitext(path)[0]
    print('Analizando: ' + get_fileName)
    subprocess.call(["./streaming_extractor_music", file, out_dir + get_fileName + ".json"])
    print('')

def extract_all_files(json_files):
    return list(map(extractor, json_files))

def integrator(file):
    with open (file) as f:
        data = json.load(f)
        datos = []
        #datos.append(data.get('lowlevel').get('barkbands_flatness_db').get('mean'))
		#datos.append(data.get('lowlevel').get('spectral_complexity').get('mean'))
        #datos.append(data.get('lowlevel').get('dynamic_complexity'))
        datos.append(data.get('lowlevel').get('mfcc').get('mean'))
		#datos.append(data.get('lowlevel').get('spectral_contrast_coeffs').get('mean'))
		#datos.append(data.get('rhythm').get('onset_rate'))
		#datos.append(0) #This is for the class in string mode
        datosFlatten = [a for x in datos for a in (x if isinstance(x, list) else [x])]
        #print('soy datos ', datos)
        #all_data.append(datos)
        all_data.append(datosFlatten)

def integrate_all_files(json_files):
    return list(map(integrator, json_files))

def save_matrix_array(file_name, matrix):
    with open(file_name, 'w') as f:
      savetxt(f, matrix)
    print('Done')

extract_all_files(sorted(glob.glob(in_dir + "*.wav")))
integrate_all_files(sorted(glob.glob(in_dir_2 + "*.json")))
save_matrix_array(file_out, all_data)