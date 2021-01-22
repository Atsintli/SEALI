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

mfccs = list(map(lambda m: m["loudness"],extract_all_mfccs(["Segments/008801.wav"])))

print("soy MFCCS",mfccs)

def tf_handler(mfccs):
  headers = {"content-type": "application/json"}
  data = {"instances": [mfccs]}
  #print ("input:",mfccs)
  r = requests.post(url = "http://localhost:8501/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
 
  data = r.json()["predictions"]
  print(data)
  return data

i=0
mfcc_results = [mfccs]

while i < 30:
    mfcc_results.append(tf_handler(mfcc_results[-1]))
    i = i+1

#with open('anotated_MSC_mfccs_name2.json') as f:
with open('anotated_MSC_loudness.json') as f:
  jsonData = json.load(f)

myData = jsonData.values()# [[{mfccMean: [], className: "1", fileName: ''}], [{mfccMean: [], className: "2", fileName: ''}]]
flatten = lambda t: [item for sublist in t for item in sublist]
myFlatData = flatten(myData) # [{mfccMean: [], className: "1", fileName: ''}, {mfccMean: [], className: "2", fileName: ''}]
#print("soy flaten", myFlatData)

def reducer( mfcc, acc, fileData):
  diff = np.linalg.norm(np.array(fileData['loudness']) - np.array(mfcc))
  #diff = spatial.distance_matrix(fileData['loudness'], np.array(mfcc))
  if acc==None:
    return tz.assoc(fileData, 'diff', diff)
  else: 
    if acc['diff'] <= diff:
        return acc
    else:
        return tz.assoc(fileData, 'diff', diff)

def getClosestCandidate(mfcc):
  return reduce(lambda acc, fileData: reducer(mfcc, acc, fileData), myFlatData, None)

result = list(map(getClosestCandidate,mfcc_results))

print (list(map(lambda x: x['file'], result)))
print(len(result))

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
                soundfile.write(folder + "test_2" + ".wav", audiototal, sr)

lista = list(map(lambda x: x['file'], result))
concatenateFiles(lista)