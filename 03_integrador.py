import sys
import json
import csv
import glob
from utils import*

rootDir = 'music18_all_features/'	
allData = []

def fix_nan_bug(files):
	with open(files, 'r') as f:
		f = f.read()
		f = f.replace("-nan",  "0")
		fixed_file = json.loads(f)
	return fixed_file

def integrador(file):
	#file_name, ext = (os.path.splitext(file))
	#file_name = file_name.split('/')[1]
	data = fix_nan_bug(file)
	lista = []
	#lista.append(file_name)
	lista.append(data['lowlevel']['mfcc']['mean'])
	lista.append(data['lowlevel']['barkbands_crest']['mean'])
	lista.append(data['lowlevel']['average_loudness'])
	lista.append(data['lowlevel']['dissonance']['mean'])
	lista.append(data['lowlevel']['dissonance']['var'])
	lista.append(data['lowlevel']['erbbands_crest']['mean'])
	lista.append(data['lowlevel']['erbbands_crest']['var'])
	lista.append(data['lowlevel']['melbands_crest']['mean'])
	lista.append(data['lowlevel']['melbands_flatness_db']['mean'])
	lista.append(data['lowlevel']['pitch_salience']['mean'])
	lista.append(data['lowlevel']['spectral_complexity']['var'])
	lista.append(data['lowlevel']['spectral_complexity']['mean'])
	lista.append(data['lowlevel']['spectral_decrease']['mean'])
	lista.append(data['lowlevel']['spectral_decrease']['var'])
	lista.append(data['lowlevel']['spectral_energy']['mean'])
	lista.append(data['lowlevel']['spectral_energy']['var'])
	lista.append(data['lowlevel']['spectral_entropy']['mean'])
	lista.append(data['lowlevel']['spectral_entropy']['var'])
	lista.append(data['lowlevel']['barkbands']['mean'])
	lista.append(data['lowlevel']['barkbands']['var'])
	lista.append(data['rhythm']['beats_count'])
	lista.append(data['rhythm']['beats_loudness']['mean'])
	lista.append(data['rhythm']['beats_loudness']['var'])
	lista.append(data['rhythm']['bpm_histogram_first_peak_bpm']['mean'])
	lista.append(data['rhythm']['danceability'])
	lista.append(data['rhythm']['beats_loudness_band_ratio']['var'])
	lista.append(data['rhythm']['beats_loudness_band_ratio']['mean'])
	lista.append(data['tonal']['chords_strength']['var'])
	lista.append(data['tonal']['chords_strength']['mean'])
	lista.append(data['tonal']['chords_histogram'])
	lista.append(data['tonal']['tuning_frequency'])
	lista.append(data['tonal']['thpcp'])
	listaFlatten = [a for x in lista for a in (x if isinstance(x, list) else [x])]
	print('escribiendo', file)
	return listaFlatten

def exec_integrador(files):
	return list(map(integrador, files))

all_features = exec_integrador(sorted(glob.glob(rootDir + "*.json")))

def save_as_matrix(features):
	save_descriptors_as_matrix('features_all.csv', features)

save_as_matrix(all_features)