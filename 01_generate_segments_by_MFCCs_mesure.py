import sys
import os
from essentia.standard import *
import numpy as np


#python3 cortador.py /Users/hugosg/Documents/OttoAudios/ /Users/hugosg/Documents/OttoSalida/
#entrada carpeta con sonidos
#salida lugar de carpeta nueva para guardar pedacitos

rootDir = '/media/atsintli/archives/audio_data_base'
out_dir = 'Segments/'
print(rootDir)
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

counter = 0

def segments_gen(fileName):
	loader = essentia.standard.MonoLoader(filename=fileName)
	# and then we actually perform the loading:
	audio = loader()

	w = Windowing(type = 'hann')
	spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
	mfcc = MFCC()
	mel = MelBands()
	loudness = Loudness()

	logNorm = UnaryOperator(type='log')
	pool = essentia.Pool()
	print("num de samples", len(audio))
	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
	    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
	    melBands = mel(spectrum(w(frame)))
	    pool.add('lowlevel.mfcc', mfcc_coeffs)
	    # pool.add('lowlevel.mfcc_bands', mfcc_bands)
	    # pool.add('lowlevel.mfcc_bands_log', logNorm(mfcc_bands))
	    pool.add('lowlevel.melbands', melBands)


	#pedazos muy grandes
	# minimumSegmentsLength = 10
	# size1 = 300
	# inc1 = 60
	# size2 = 200
	# inc2 = 60
	# cpw = 1.5

		#pedazos muy pequenos
	minimumSegmentsLength = 10
	size1 = 300
	inc1 = 60
	size2 = 200
	inc2 = 60
	cpw = 9.5

	features = [val for val in pool['lowlevel.mfcc'].transpose()]
	#features = [val for val in pool['lowlevel.melbands'].transpose()]
	#print(features)
	sbic = SBic(size1=size1, inc1=inc1,size2=size2, inc2=inc2,cpw=cpw, minLength=minimumSegmentsLength)
	segments = sbic(np.array(features))
	#print(segments)
	record_segments(audio,segments)

def record_segments(audio, segments):
	for segment_index in range(len(segments) - 1):
		global counter
		start_position = int(segments[segment_index] * 512)
		end_position = int(segments[segment_index + 1] * 512)
		writer = essentia.standard.MonoWriter(filename=out_dir + "{:05d}".format(counter) + ".wav", format="wav")(audio[start_position:end_position])
		counter = counter + 1

for root, dirs, files in os.walk(rootDir):
    path = root.split(os.sep)
    for file in files:
        #print(len(path) * '---', file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() == ".wav":
        	print('Cuting: ', file)
        	segments_gen(root + "/" + file)