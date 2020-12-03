from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import sys
import os
from essentia.standard import *
import numpy as np
import glob
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default


in_dir = 'audiotest/'
out_dir = 'segmentsTest_erase/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

counter = 0

segmentos = []

def segments_gen(fileName):
    loader = essentia.standard.MonoLoader(filename=fileName, sampleRate=44100)
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
    for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512, startFromZero=True): #con 44100 fr=2048, hs=512
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
    minimumSegmentsLength = 10
    size1 = 300
    inc1 = 60
    size2 = 200
    inc2 = 60
    cpw = 4.5

    features = [val for val in pool['lowlevel.mfcc'].transpose()]
    #features = [val for val in pool['lowlevel.melbands'].transpose()]
    sbic = SBic(size1=size1, inc1=inc1,size2=size2, inc2=inc2,cpw=cpw, minLength=minimumSegmentsLength)
    segments = sbic(np.array(features))
    record_segments(audio,segments)
    segmentos.append(segments.astype(int))
    plot(fileName)
    return segmentos

def plot(fileName):
    sample_rate=44100
    y, sr = librosa.load(fileName, sr=sample_rate)
    #librosa.onset.onset_detect(y=y, sr=sr, units='time')
    o_env = librosa.onset.onset_strength(y, sr=sr)
    times = librosa.times_like(o_env, sr=sr)

    #onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    D = np.abs(librosa.stft(y, n_fft=2048*8))

    fig, ax = plt.subplots(nrows=2, sharex=True)

    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                            x_axis='time', y_axis='log', ax=ax[0], sr=sr, 
                            hop_length=4096)

    ax[0].set(title='Power spectrogram')

    ax[0].label_outer()

    ax[1].plot(times, o_env, label='Onset strength')
    print("hola soy segmentos ", segmentos)
    ax[1].vlines(times[segmentos], 0, o_env.max(), color='black', alpha=0.4,

            linestyle='--', label='Sbic Segments')

    ax[1].legend()
    plt.show()

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

input_data = gen_all_segments(sorted(glob.glob(in_dir + "*.wav")))

print("Done")






