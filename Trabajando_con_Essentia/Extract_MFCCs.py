import essentia
import essentia.standard as ess
from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
import csv

def extract_mfccs (audio):
    loader = essentia.standard.MonoLoader(filename=audio)
    audio = loader()
    mfcc = MFCC(numberCoefficients=13)
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    w = Windowing(type = 'hann')

    pool = essentia.Pool()
    for frame in ess.FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        #pool.add('lowlevel.mfcc_bands', mfcc_bands)
        #pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))
    
    #YamlOutput(filename = 'mfcc.sig', format='yaml', writeVersion=False)(pool)

    # compute mean and variance of the frames
    #aggrPool = PoolAggregator(defaultStats = [ 'mean', 'stdev' ])(pool)
    aggrPool = PoolAggregator(defaultStats = ['mean'])(pool)

    # and ouput those results in a file
    YamlOutput(filename = 'mfccmean.sig', format='yaml', writeVersion=False)(aggrPool)

    file = open('mfccmean.sig').read()
    mfccs = file[35:]
    m = np.matrix(mfccs)
    print(mfccs)
    savetxt(f, m)

file_name = 'mfccs.csv'
f = open(file_name, 'w')

for audio_files in sorted(glob.glob( 'Segments/' + "*.wav" )):
    print(audio_files)
    mfccs = extract_mfccs(audio_files)