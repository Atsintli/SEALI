import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack
#from IPython import display
from essentia.streaming import *
from essentia import Pool, run, array, reset
from scipy.special import softmax
from essentia import INFO
from feature_extract import *
from functools import reduce
from toolz import assoc
#OSC libs
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

input_data = extract_all_mfccs(sorted(glob.glob('Segments/' + "*.wav")[22000:22010]))
#print(input_data)
mfccs = list(map(lambda m: m["mfccMean"],input_data))
#print(mfccs)

#print("soy MFCCS",mfccs)

def tf_handler(mfccs):
  headers = {"content-type": "application/json"}
  data = {"instances": [mfccs]}
  #print ("input:",mfccs)
  r = requests.post(url = "http://localhost:8501/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  response = r.json()
  #print(response)
  data = response["predictions"]
  #print(data)
  return data[0]

i=0
mfcc_results = mfccs

while i < 200:
    data_result = mfcc_results[-9:]
    print("new_MFCCS", data_result)
    mfcc_results.append(tf_handler(mfcc_results[-9:]))
    i = i+1

with open('anotated_MSC_mfccs_name2.json') as f:
  jsonData = json.load(f)

myData = jsonData.values()# [[{mfccMean: [], className: "1", fileName: ''}], [{mfccMean: [], className: "2", fileName: ''}]]
flatten = lambda t: [item for sublist in t for item in sublist]
myFlatData = flatten(myData) # [{mfccMean: [], className: "1", fileName: ''}, {mfccMean: [], className: "2", fileName: ''}]
#print("soy flaten", myFlatData)

def reducer( mfcc, acc, fileData):
  diff = np.linalg.norm(np.array(fileData['mfccMean']) - np.array(mfcc))
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
            num = ('Segments/' + elements)
            for audio_files in sorted(glob.glob(num + "*.wav")):
                print("Escribiendo " + audio_files)
                y, sr = librosa.load(audio_files)
                audiototal = np.append(audiototal, y)
                soundfile.write(folder + "test_mfccs_2" + ".wav", audiototal, sr)

lastResult = dedupe(result)
lista = list(map(lambda x: x['file'], lastResult))
concatenateFiles(lista)