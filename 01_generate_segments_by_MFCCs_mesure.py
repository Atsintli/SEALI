import sys
import os
from essentia.standard import *
import numpy as np
import glob
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default


#in_dir = '../../Samsung/Música/aarón/SEALI_HP/04/'
in_dir = '../../Samsung/Música/Music_18/'
out_dir = 'segments_music18/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

counter = 0

def segments_gen(fileName):
    loader = essentia.standard.MonoLoader(filename=fileName)
    audio = loader()
    print('\n')
    print("Generating Segments: " + fileName)
    print("Num of samples: ", len(audio))

    w = Windowing(type = 'hann')
    spectrum = Spectrum()
    mfcc = MFCC()
    #mel = MelBands()
    #loudness = Loudness()

    logNorm = UnaryOperator(type='log')
    pool = essentia.Pool()
    for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512, startFromZero=True): #2048, 512
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        #melBands = mel(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        #pool.add('lowlevel.mfcc_bands', mfcc_bands)
        #pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))
        #pool.add('lowlevel.melbands', melBands)

    #Sbic Parameters

    #segmentos grandes
    # minimumSegmentsLength = 10
    # size1 = 300
    # inc1 = 60
    # size2 = 200
    # inc2 = 60
    # cpw = 1.5

    #segmentos medianos
    # minimumSegmentsLength = 10
    # size1 = 300
    # inc1 = 60
    # size2 = 200
    # inc2 = 60
    # cpw = 4.5

    # #segmentos diminutos
    # minimumSegmentsLength = 1
    # size1 = 50
    # inc1 = 30
    # size2 = 50
    # inc2 = 30
    # cpw = 1

    #segmentos diminutos
    minimumSegmentsLength = 1
    size1 = 10
    inc1 = 10
    size2 = 10
    inc2 = 10
    cpw = 1

    features = [val for val in pool['lowlevel.mfcc'].transpose()]
    #features = [val for val in pool['lowlevel.melbands'].transpose()]
    sbic = SBic(size1=size1, inc1=inc1,size2=size2, inc2=inc2,cpw=cpw, minLength=minimumSegmentsLength)
    segments = sbic(np.array(features))
    record_segments(audio,segments)

    # plot(audio)
    # for segment in segments:
    #     plt.axvline(x=segment*512, color='black')
    # plt.title("Audio waveform and the estimated onset positions (HFC onset detection function)")
    # show()

def record_segments(audio, segments):
    for segment_index in range(len(segments) - 1):
        global counter
        start_position = int(segments[segment_index] * 512)
        end_position = int(segments[segment_index + 1] * 512)
        writer = essentia.standard.MonoWriter(filename=out_dir + "{:06d}".format(counter) + ".wav", format="wav")(audio[start_position:end_position])
        counter = counter + 1
    print('Num of Segments: ' + str(len(segments)))

def gen_all_segments(audio_files):
	return list(map(segments_gen, audio_files))

input_data = gen_all_segments(sorted(glob.glob(in_dir + "*.mp3")))

print("Done")
