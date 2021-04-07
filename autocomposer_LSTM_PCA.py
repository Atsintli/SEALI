import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack
from essentia.streaming import *
from essentia import Pool, run, array, reset
from scipy.special import softmax
from essentia import INFO
from feature_extract import *
from functools import reduce
from toolz import assoc
import argparse
import math
import requests # importing the requests library 
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import json
from utils import *
import soundfile
from scipy import spatial
from sklearn.decomposition import PCA   
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa

input_data = extract_all_mfccs(sorted(glob.glob('segments_music18/' + "*.wav")[90:100]))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data: 
               list(tz.concat(getProps(
                ['flatness', 'mfccVar','complexity','mfccMean','loudness','centroid','spectralContrast'],
                   data))),
    input_data))
    np_mfcc = np.array(features)
    #print(np_mfcc.shape)

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(np_mfcc)
    pca = PCA(n_components=0.4, whiten=True)
    pca_result = pca.fit_transform(x_train)
    #print(pca_result.shape)
    pca_result = pca_result.tolist()
    return pca_result

def tf_handler(mfccs):
  headers = {"content-type": "application/json"}
  data = {"instances": [mfccs]}
  #print ("input:",mfccs)
  r = requests.post(url = "http://localhost:8501/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  response = r.json()
  #print(response)
  data = response["predictions"]
  #print("soy data",data)
  return data[0]

i=0
mfccs = concat_features(input_data)
mfcc_results = mfccs
#print("deben ser pca",mfcc_results[-9:])

while i < 10:
    data_result = mfcc_results[-9:]
    print("new_MFCCS", data_result)
    mfcc_results.append(tf_handler(mfcc_results[-9:])) #add data(prediction) to mfcc_results
    #print("soy mfccresults", mfcc_results)
    i = i+1

with open('pca_music18.json') as f:
  jsonData = json.load(f)

def reducer( mfcc, acc, fileData):
  diff = np.linalg.norm(np.array(fileData['PCA']) - np.array(mfcc))
  #print("llego aqui!!!!", type(mfcc))
  #print (fileData['mfccMean']+fileData['mfccVar'])
  #diff = spatial.distance_matrix(np.array(fileData['mfccMean']), np.array(mfcc))
  if acc==None:
    return tz.assoc(fileData, 'diff', diff)
  else: 
    if acc['diff'] <= diff:
        return acc
    else:
        return tz.assoc(fileData, 'diff', diff)

def getClosestCandidate(mfcc):
  return reduce(lambda acc, fileData: reducer(mfcc, acc, fileData), jsonData, None)

result = list(map(getClosestCandidate,mfcc_results))[10:]

print (list(map(lambda x: x['file'], result)))
print(len(result))

def dedupe(tracklist):
    acc = []
    for el in tracklist:
        if len(acc) == 0:
            acc.append(el)
            continue;
        if acc[-1]["file"] != el["file"]:
            acc.append(el)
    return acc

def concatenateFiles(fileName):
    folder = 'Autocomposer/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    files = []

    for i in range(1):
        audiototal = np.array([])
        for elements in fileName:
            num = ('segments_music18/' + elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                soundfile.write(folder + "music18_2" + ".wav", audiototal, sr)

print("result", result)
lastResult = dedupe(result)
lista = list(map(lambda x: x['file'], lastResult))
print("last list", lista)
concatenateFiles(lista)