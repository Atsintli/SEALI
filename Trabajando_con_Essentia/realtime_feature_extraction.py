import numpy as np
import matplotlib.pyplot as plt
import soundcard as sc
from struct import unpack
from IPython import display
from essentia.streaming import *
from essentia import Pool, run, array, reset
from scipy.special import softmax

sampleRate = 16000
frameSize = 512 
hopSize = 256
numberBands = 13

# analysis parameters
patchSize = 64 #control of velocity of the extractor
displaySize = 10

bufferSize = patchSize * hopSize
buffer = np.zeros(bufferSize, dtype='float32')
vectorInput = VectorInput(buffer)
frameCutter = FrameCutter(frameSize=frameSize, hopSize=hopSize)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC()

pool = Pool()

vectorInput.data   >> frameCutter.signal
frameCutter.frame  >>  w.frame >> spec.frame 
spec.spectrum >> mfcc.spectrum
mfcc.bands >> None
mfcc.mfcc >> (pool, 'lowlevel.mfcc.mean')

def callback(data):
    # update audio buffer
    buffer[:] = array(unpack('f' * bufferSize, data))
    print ("this is the buffer", buffer[:])
    
    #mfccBuffer = np.zeros([numberBands, patchSize * displaySize])
    mfccBuffer = np.zeros([numberBands])
    print ("np zeros",mfccBuffer)

    # generate predictions
    reset(vectorInput)
    run(vectorInput)
    print('Pool contains %d frames of MFCCs' % len(pool['lowlevel.mfcc.mean']))
    print (buffer)

    # update mel and activation buffers
    mfccBuffer = np.roll(mfccBuffer, -patchSize)
    mfccBuffer = pool['lowlevel.mfcc.mean'][-patchSize].T
    print ("mfccs", mfccBuffer)

    return mfccBuffer

#reset storage
pool.clear()

# capture and process the speakers loopback
# the 2 selects the external interface Zoom h5
with sc.all_microphones(include_loopback=True)[2].recorder(samplerate=sampleRate) as mic:
    while True:
        callback(mic.record(numframes=bufferSize).mean(axis=1))