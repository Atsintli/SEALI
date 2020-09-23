import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack
#from IPython import display
from essentia.streaming import *
from essentia import Pool, run, array, reset
from scipy.special import softmax
from essentia import INFO

#OSC libs
import argparse
import math
import requests # importing the requests library 
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import json

sampleRate = 16000
frameSize = 2048 
hopSize = 2048
numberBands = 13
onsets = 1

# analysis parameters
patchSize = 20  #control the velocity of the extractor 60 is approximately one second of audio
displaySize = 10

bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
vectorInput = VectorInput(buffer)
frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC(numberCoefficients=13)
#loudness = Loudness()
fft = FFT() # this gives us a complex FFT
c2p = CartesianToPolar()
onset = OnsetDetection()

pool = Pool()

vectorInput.data  >> frameCutter.signal
frameCutter.frame >> w.frame >> spec.frame
spec.spectrum     >> mfcc.spectrum
mfcc.bands        >> None
mfcc.mfcc         >> (pool, 'mfcc')
w.frame           >> fft.frame
fft.fft           >> c2p.complex
c2p.magnitude     >> onset.spectrum
c2p.phase         >> onset.phase
onset.onsetDetection >> (pool, 'onset')

def callback(data):
    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))
    #print ("this is the buffer", buffer[:])
    mfccBuffer = np.zeros([numberBands])
    onsetBuffer = np.zeros([onsets])
    reset(vectorInput)
    run(vectorInput)
    mfccBuffer = np.roll(mfccBuffer, -patchSize)
    onsetBuffer = np.roll(mfccBuffer, -patchSize)
    mfccBuffer = pool['mfcc'][-patchSize]
    onsetBuffer = pool['onset'][-patchSize]
    print ("MFCCs:", '\n', (mfccBuffer))
    print ("OnsetDetection:", '\n', onsetBuffer)
    features = np.concatenate((mfccBuffer, onsetBuffer), axis=None)
    features = features.tolist()
    return features

def tf_handler(args):
  headers = {"content-type": "application/json"}
  data = {"instances": [args]}
  r = requests.post(url = "http://localhost:8501/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  data = r.json()["predictions"]
  
  clase_0 = data[0][0]
  clase_1 = data[0][1]
  clase_2 = data[0][2]
  clase_3 = data[0][3]
  clase_4 = data[0][4]
  clase_5 = data[0][5]
  # clase_6 = data[0][6]
  # clase_7 = data[0][3]

  event = max([clase_0,clase_1,clase_2,clase_3,clase_4,clase_5])
  if event == clase_0:
    print (event, "\t", "clase_0")
  if event == clase_1:
    print (event, "\t", "clase_1")
  if event == clase_2:
    print (event, "\t", "clase_2")
  if event == clase_3:
    print (event, "\t", "clase_3")
  if event == clase_4:
    print (event, "\t", "clase_4")
  if event == clase_5:
    print (event, "\t", "clase_5")
  # if event == clase_7:
  #   print (event, "clase_7")
  # if event == clase_8:
  #   print (event, "clase_8")

  # printable_data = "Compuesto", str(data)
  # if clase_0 > 0.4:
  #   printable_data = "clase_0", str(clase_0)
  # if clase_1 > 0.4:
  #   printable_data = "clase_1", str(clase_1)
  # if clase_2 > 0.4:
  #   printable_data = "clase_2", str(clase_2)
  # if clase_3 > 0.4:
  #   printable_data = "clase_3", str(clase_3)
  # if clase_4 > 0.4:
  #   printable_data = "clase_4", str(clase_4)
  # if clase_5 > 0.4:
  #   printable_data = "clase_5", str(clase_5)
  # if clase_6 > 0.4:
  #   printable_data = "clase_6", str(clase_6)
  # print('\n', printable_data)
  print ('\t', "Prediction:", data)

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5 #3 for jack
with sc.all_microphones(include_loopback=True)[3].recorder(samplerate=sampleRate) as mic:
  while True:
    tf_handler(callback(mic.record(numframes=bufferSize).mean(axis=1)) )
    #print ('\n', prediction)
