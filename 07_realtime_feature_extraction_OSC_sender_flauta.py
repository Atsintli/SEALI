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

client = udp_client.SimpleUDPClient('127.0.0.2', 5008) #this client sends to SC

sampleRate = 44100
frameSize = 2048
hopSize = 2048
numberBands = 3
onsets = 1
loudness = 1
patchSize = 60  #control the velocity of the extractor 20 is approximately one second of audio
displaySize = 10

bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
vectorInput = VectorInput(buffer)
frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC(numberCoefficients=13)

pool = Pool()

vectorInput.data  >> frameCutter.signal
frameCutter.frame >> w.frame >> spec.frame
spec.spectrum     >> mfcc.spectrum
mfcc.bands        >> None
mfcc.mfcc         >> (pool, 'mfcc')

def callback(data):
    buffer[:] = array(unpack('f' * bufferSize, data))
    mfccBuffer = np.zeros([numberBands])

    reset(vectorInput)
    run(vectorInput)
    mfccBuffer = np.roll(mfccBuffer, -patchSize)

    mfccBuffer = pool['mfcc'][-patchSize]

    features = mfccBuffer
    features = features.tolist()
    print(features)
    return features

def tf_handler(args):
  headers = {"content-type": "application/json"}
  data = {"instances": [args]}
  r = requests.post(url = "http://localhost:8502/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  response = r.json()
  data = response["predictions"]
  print(data)
  client.send_message("/clase", *data)

  clases=data[0]
  event = max(clases)
  index = clases.index(event)
  print ("Clase Predominante", index)

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5 #3 for jack
with sc.all_microphones(include_loopback=True)[3].recorder(samplerate=sampleRate) as mic:
  while True:
    tf_handler(callback(mic.record(numframes=bufferSize).mean(axis=1)) )
    #print ('\n', prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
    default="127.0.0.2", help="The ip to listen on")
    parser.add_argument("--port",
    type=int, default=5007, help="The port to listen on")
    args = parser.parse_args()
    #dispatcher = dispatcher.Dispatcher()
    #dispatcher.map("/features", tf_handler)
    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
