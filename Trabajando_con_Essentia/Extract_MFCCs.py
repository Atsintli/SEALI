#### this reproduces the way inverse DCT is applied on MFCC to convert back to spectral domain. 
#### following the implementation:  http://labrosa.ee.columbia.edu/matlab/rastamat/ 

import essentia
import essentia.standard as ess
from essentia.streaming import *
#from essentia.standard import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import glob
# pylab contains the plot() function, as well as figure, etc... (same names as Matlab)
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

#rewriting the same file
file_name = 'features.csv'
file = open(file_name, 'w')

""" def feature_extractor(filename):    
    frameSize = 1024
    hopSize = 512
    fs = 44100
    audio = ess.MonoLoader(filename=filename, sampleRate=fs)()
    w = ess.Windowing(type='hamming', normalized=False)
    # make sure these are same for MFCC and IDCT computation
    NUM_BANDS  = 26
    DCT_TYPE = 2
    LIFTERING = 0
    NUM_MFCCs = 13

    spectrum = ess.Spectrum()
    mfcc = ess.MFCC(numberBands=NUM_BANDS,
                    numberCoefficients=NUM_MFCCs, # make sure you specify first N mfcc: the less, the more lossy (blurry) the smoothed mel spectrum will be
                    weighting='linear', # computation of filter weights done in Hz domain (optional)
                    normalize='unit_max', #  htk filter normaliation to have constant height = 1 (optional)
                    dctType=DCT_TYPE,
                    logType='log',
                    liftering=LIFTERING) # corresponds to htk default CEPLIFTER = 22

    idct = ess.IDCT(inputSize=NUM_MFCCs, 
                    outputSize=NUM_BANDS, 
                    dctType = DCT_TYPE, 
                    liftering = LIFTERING)

    all_melbands_smoothed = []

    for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        spect = spectrum(w(frame))
        melbands, mfcc_coeffs = mfcc(spect)
        melbands_smoothed = np.exp(idct(mfcc_coeffs)) # inverse the log taken in MFCC computation
        all_melbands_smoothed.append(melbands_smoothed)
    
    print(all_melbands_smoothed)

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    # mfccs = essentia.array(pool['MFCC']).T
    all_melbands_smoothed = essentia.array(all_melbands_smoothed).T

    # and plot
    #plt.imshow(all_melbands_smoothed, aspect='auto', interpolation='none') # ignore enery
    # plt.imshow(mfccs, aspect = 'auto', interpolation='none')
    #plt.show() # unnecessary if you started "ipython --pylab"
    
    features = np.hstack([all_melbands_smoothed])
    b = np.matrix(features)
    savetxt(file, b)


for audio_files in glob.glob( 'audio/1/' + "*.wav" ):
  feature_extractor(audio_files)
from essentia.standard import *
w = Windowing(type = 'hann')
spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
mfcc = MFCC()

# we start by instantiating the audio loader:
loader = essentia.standard.MonoLoader(filename='0200_segment.wav')

# and then we actually perform the loading:
audio = loader()

frame = audio[6*44100 : 6*44100 + 1024]
spec = spectrum(w(frame))
mfcc_bands, mfcc_coeffs = mfcc(spec)
logNorm = UnaryOperator(type='log')
# plot(spec)
# plt.title("The spectrum of a frame:")
# show()

# plot(mfcc_bands)
# plt.title("Mel band spectral energies of a frame:")
# show()

# plot(mfcc_coeffs)
# plt.title("First 13 MFCCs of a frame:")
# show()

# mfccs = []
# melbands = []
# melbands_log = []

# for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
#     mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
#     mfccs.append(mfcc_coeffs)
#     melbands.append(mfcc_bands)
#     melbands_log.append(logNorm(mfcc_bands))

# # transpose to have it in a better shape
# # we need to convert the list to an essentia.array first (== numpy.array of floats)
# mfccs = essentia.array(mfccs).T
# melbands = essentia.array(melbands).T
# melbands_log = essentia.array(melbands_log).T

# # and plot
# imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
# plt.title("Mel band spectral energies in frames")
# show()

# imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
# plt.title("Log-normalized mel band spectral energies in frames")
# show()

# imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
# plt.title("MFCCs in frames")
# show()
'''

##########################POOL###########################

pool = essentia.Pool()

for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    pool.add('lowlevel.mfcc', mfcc_coeffs)
    #pool.add('lowlevel.mfcc_bands', mfcc_bands)
    #pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))

# imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', origin='lower', interpolation='none')
# plt.title("Mel band spectral energies in frames")
# show()

# imshow(pool['lowlevel.mfcc_bands_log'].T, aspect = 'auto', origin='lower', interpolation='none')
# plt.title("Log-normalized mel band spectral energies in frames")
# show()

# imshow(pool['lowlevel.mfcc'].T[1:,:], aspect='auto', origin='lower', interpolation='none')
# plt.title("MFCCs in frames")
# show()

# or as a one-liner:
#YamlOutput(filename = 'mfcc.sig')(pool)

# compute mean and variance of the frames
aggrPool = PoolAggregator(defaultStats = ['mean'])(pool)

# and ouput those results in a file
YamlOutput(filename = 'mfccaggr.sig')(aggrPool)
'''

#data = open('mfccaggr.sig', 'r')
#data = data.read
#print(data)

mfcc.mfcc.disconnect((pool, 'lowlevel.mfcc'))
"""

loader = MonoLoader(filename = '0200_segment.wav')
frameCutter = FrameCutter(frameSize = 1024, hopSize = 512)
w = Windowing(type = 'hann')
spec = Spectrum()
mfcc = MFCC()

loader.audio >> frameCutter.signal
frameCutter.frame >> w.frame >> spec.frame
spec.spectrum >> mfcc.spectrum

#pool = essentia.Pool()

mfcc.bands >> None
#mfcc.mfcc >> (pool, 'lowlevel.mfcc.mean')

#essentia.run(loader)

#print('Pool contains %d frames of MFCCs' % len(pool['lowlevel.mfcc.mean']))

#mfcc.mfcc.disconnect((pool, 'lowlevel.mfcc'))

fileout = FileOutput(filename = 'mfccframes.txt')
#mfcc.mfcc >> (pool, 'lowlevel.mfcc.mean')
mfcc.mfcc >> fileout

essentia.reset(loader)
essentia.run(loader)


