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

client = udp_client.SimpleUDPClient('127.0.0.1', 5006) #this client sends to SC

sampleRate = 44100
frameSize = 2048 
hopSize = 2048
numberBands = 3
onsets = 1
loudness = 1

# analysis parameters
patchSize = 60  #control the velocity of the extractor 20 is approximately one second of audio
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
eqloud = EqualLoudness() #checar esto!!!

pool = Pool()
#b = LoudnessEBUR128(hopSize=0.1, sampleRate=44100)

vectorInput.data  >> eqloud.signal >> frameCutter.signal
frameCutter.frame >> w.frame >> spec.frame
spec.spectrum     >> mfcc.spectrum
mfcc.bands        >> None
mfcc.mfcc         >> (pool, 'mfcc')
w.frame           >> fft.frame
fft.fft           >> c2p.complex
c2p.magnitude     >> onset.spectrum
c2p.phase         >> onset.phase
#b.momentaryLoudness >> (pool, 'momentaryLoudness')
onset.onsetDetection >> (pool, 'onset')

def callback(data):
    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))
    #print ("this is the buffer", buffer[:])
    mfccBuffer = np.zeros([numberBands])
    #onsetBuffer = np.zeros([onsets])
    #loudnessBuffer = np.zeros([loudness])
    reset(vectorInput)
    run(vectorInput)
    mfccBuffer = np.roll(mfccBuffer, -patchSize)
    #onsetBuffer = np.roll(mfccBuffer, -patchSize)
    #loudnessBuffer = np.roll(loudnessBuffer, -patchSize)
    mfccBuffer = pool['mfcc'][-patchSize]
    #onsetBuffer = pool['onset'][-patchSize]
    #loudnessBuffer = pool['momentaryLoudness'][-patchSize]
    #print ("MFCCs:", '\n', (mfccBuffer))
    #print ("OnsetDetection:", '\n', onsetBuffer)
    #print ("momentaryLoudness:", '\n', loudnessBuffer)
    #features = np.concatenate((mfccBuffer, onsetBuffer), axis=None)
    features = mfccBuffer
    features = features.tolist()
    print(features)
    return features

def tf_handler(args):
  headers = {"content-type": "application/json"}
  data = {"instances": [args]}
  #data = {"instances": [[-1264.91162109375, 3.0517578125e-05, -6.866455078125e-05]]} #this format is correct
  #print([[*args]])
  r = requests.post(url = "http://localhost:8501/v1/models/improv_class:predict", data=json.dumps(data), headers=headers)
  #data = r.json()["predictions"]
  response = r.json()
  #print(response)
  data = response["predictions"]
  print(data)
  #return data[0]
  client.send_message("/clase", *data)

  clases=data[0]
  event = max(clases)
  index = clases.index(event)
  print ("Clase Predominante", index)
  
  # clase_0 = data[0][0]
  # clase_1 = data[0][1]
  # clase_2 = data[0][2]
  # clase_3 = data[0][3]
  # clase_4 = data[0][4]
  #clase_5 = data[0][5]
  #clase_6 = data[0][6]
  # clase_7 = data[0][7]
  # clase_8 = data[0][8]
  # clase_9 = data[0][9]

  #event = max([clase_0,clase_1,clase_2,clase_3,clase_4,
  #clase_4,clase_5,clase_6
  #,clase_7,clase_8,clase_9
  #])

  # if event == clase_0:
  #   print (event, "\t", "clase_0")
  # if event == clase_1:
  #   print (event, "\t", "clase_1")
  # if event == clase_2:
  #   print (event, "\t", "clase_2")
  # if event == clase_3:
  #   print (event, "\t", "clase_3")
  # if event == clase_4:
  #   print (event, "\t", "clase_4")
  #if event == clase_5:
  #  print (event, "\t", "clase_5")
  #if event == clase_6:
  #   print (event, "\t", "clase_6")
  # if event == clase_7:
  #    print (event, "\t", "clase_7")
  # if event == clase_8:
  #    print (event, "\t", "clase_8")
  # if event == clase_9:
  #    print (event, "\t", "clase_9")

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
  #print ('\t', "Prediction:", data)

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5 #3 for jack
with sc.all_microphones(include_loopback=True)[3].recorder(samplerate=sampleRate) as mic:
  while True:
    tf_handler(callback(mic.record(numframes=bufferSize).mean(axis=1)) )
    #print ('\n', prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
    default="127.0.0.1", help="The ip to listen on") 
    parser.add_argument("--port",
    type=int, default=5005, help="The port to listen on") 
    args = parser.parse_args()
    #dispatcher = dispatcher.Dispatcher()
    #dispatcher.map("/features", tf_handler)
    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()