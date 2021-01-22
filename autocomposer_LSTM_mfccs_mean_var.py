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

input_data = extract_all_mfccs(sorted(glob.glob('segments_short/' + "*.wav")[0:10]))

def getProps(props, dict):
    return map(lambda prop: dict[prop], props)

def concat_features(input_data):
    features = list(map(lambda data: 
               list(tz.concat(getProps(
                   ['mfccMean', 'mfccVar'],
                   data))),
    input_data))
    return features


def tf_handler(mfccs):
  headers = {"content-type": "application/json"}
  data = {"instances": [mfccs]}
  #print ("input:",mfccs)
  r = requests.post(url = "http://localhost:8501/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  response = r.json()
  #print(response)
  data = response["predictions"]
  print(data)
  return data[0]

i=0
mfccs = concat_features(input_data)
mfcc_results = mfccs

while i < 5:
    data_result = mfcc_results[-9:]
    print("new_MFCCS", data_result)
    mfcc_results.append(tf_handler(mfcc_results[-9:]))
    i = i+1

with open('anotated_mfcc_mean_var.json') as f:
  jsonData = json.load(f)

myData = jsonData.values()# [[{mfccMean: [], className: "1", fileName: ''}], [{mfccMean: [], className: "2", fileName: ''}]]
flatten = lambda t: [item for sublist in t for item in sublist]
myFlatData = flatten(myData) # [{mfccMean: [], className: "1", fileName: ''}, {mfccMean: [], className: "2", fileName: ''}]
#print("soy flaten", myFlatData)

def reducer( mfcc, acc, fileData):
  diff = np.linalg.norm(np.array(fileData['mfccMean']+fileData['mfccVar']) - np.array(mfcc))
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
  return reduce(lambda acc, fileData: reducer(mfcc, acc, fileData), myFlatData, None)

result = list(map(getClosestCandidate,mfcc_results))[10:]

#print (list(map(lambda x: x['file'], result)))
#print(len(result))

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
            num = ('segments_short/' + elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                soundfile.write(folder + "test_mfccs_mean_var" + ".wav", audiototal, sr)

lastResult = dedupe(result)
lista = list(map(lambda x: x['file'], lastResult))
concatenateFiles(lista)