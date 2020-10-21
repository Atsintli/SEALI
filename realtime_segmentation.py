#Realtime segmentation
import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack
from essentia.streaming import * 
from essentia import Pool, run, array, reset, INFO
from essentia.standard import SBic
from scipy.special import softmax
import os

out_dir = 'realtime_Segmentation/'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

sampleRate = 44100
frameSize = 2048 
hopSize = 2048
numberBands = 13
onsets = 1

# analysis parameters
patchSize = 100  #control the velocity of the extractor 20 is approximately one second of audio
displaySize = 10

#sbic parameters
minimumSegmentsLength = 10
size1 = 300
inc1 = 60
size2 = 200
inc2 = 60
cpw = 9.5

bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
vectorInput = VectorInput(buffer)
frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC(numberCoefficients=13)
eqloud = EqualLoudness()
sbic = SBic(size1=size1, inc1=inc1,size2=size2, inc2=inc2,cpw=cpw, minLength=minimumSegmentsLength)
pool = Pool()

vectorInput.data  >> eqloud.signal >> frameCutter.signal
frameCutter.frame >> w.frame >> spec.frame
spec.spectrum     >> mfcc.spectrum
mfcc.bands        >> None
mfcc.mfcc         >> (pool, 'mfcc')

def callback(data):
    buffer[:] = array(unpack('f' * bufferSize, data))
    #print ("this is the buffer", buffer[:])
    mfccBuffer = np.zeros([numberBands])
    reset(vectorInput)
    run(vectorInput)
    mfccBuffer = np.roll(mfccBuffer, -patchSize)
    mfccBuffer = pool['mfcc'][-patchSize]
    features = mfccBuffer
    features = [val for val in pool['mfcc'].transpose()]
    segments = sbic(np.array(features))
    record_segments(buffer,segments)
    return features

counter = 0

def record_segments(audio, segments):
	for segment_index in range(len(segments) - 1):
		global counter
		start_position = int(segments[segment_index] * 512)
		end_position = int(segments[segment_index + 1] * 512)
		writer = essentia.standard.MonoWriter(filename=out_dir + "{:06d}".format(counter) + ".wav", format="wav")(audio[start_position:end_position])
		counter = counter + 1

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5 #3 for jack
with sc.all_microphones(include_loopback=True)[0].recorder(samplerate=sampleRate) as mic:
  while True:
      callback(mic.record(numframes=bufferSize).mean(axis=1))
    #print ('\n', prediction)
