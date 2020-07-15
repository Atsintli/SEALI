import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack
from IPython import display
from essentia.streaming import *
from essentia import Pool, run, array, reset
from scipy.special import softmax

#OSC libs
import argparse
import math
import requests # importing the requests library 
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import json

sampleRate = 16000
frameSize = 512 
hopSize = 256
numberBands = 13

# analysis parameters
patchSize = 66 #control of velocity of the extractor 60 is approximately one second of audio
displaySize = 10

bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
vectorInput = VectorInput(buffer)
frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC(numberCoefficients=19)
loudness = Loudness()

pool = Pool()

vectorInput.data  >> frameCutter.signal
frameCutter.frame >>  w.frame >> spec.frame 
spec.spectrum     >> mfcc.spectrum
mfcc.bands        >> None
mfcc.mfcc         >> (pool, 'lowlevel.mfcc.mean')

# falta agregar loudness!!!
#print((frameCutter))
#frameCutter.frame  >>  w.frame >> spec.frame 
#vectorInput.data  >> loudness.signal >> (pool, 'lowlevel.loudness')

def callback(data):
    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))
    #print ("this is the buffer", buffer[:])
    mfccBuffer = np.zeros([numberBands])
    reset(vectorInput)
    run(vectorInput)
    #print('Pool contains %d frames of MFCCs' % len(pool['lowlevel.mfcc.mean']))
    mfccBuffer = np.roll(mfccBuffer, -patchSize)
    mfccBuffer = pool['lowlevel.mfcc.mean'][-patchSize].T
    print ("MFCCs:", '\n', mfccBuffer)
    mfccBufferList = []
    mfccBufferList.append(mfccBuffer.tolist())
    return mfccBufferList

def tf_handler(args):
  headers = {"content-type": "application/json"}
  data = {"instances": [*args]}
  r = requests.post(url = "http://localhost:8501/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  data = r.json()
  print ('\n', data)

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5
with sc.all_microphones(include_loopback=True)[2].recorder(samplerate=sampleRate) as mic:
  while True:
    tf_handler(callback(mic.record(numframes=bufferSize).mean(axis=1)) )
    #print ('\n', prediction)
