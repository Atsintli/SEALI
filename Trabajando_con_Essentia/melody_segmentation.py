from pylab import plot, show, figure, imshow
#'exec(%matplotlib inline)'
import matplotlib.pyplot as plt
import numpy as np
import essentia
import essentia.standard as es
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default
from essentia.standard import *
import numpy
import os

###############################Melody detection#################################

# Load audio file; it is recommended to apply equal-loudness filter for PredominantPitchMelodia
sr = 44100
hS = 254
fS=2048

audio = 'audio/0/dumitrescu.wav'
file = audio.split('/')[2]
audio = MonoLoader(filename=audio)()

# print("Number of frames:", len(audio))
# print("Duration of the audio sample [sec]:", len(audio)/sr)
# print ("Vector real:", audio.size)
#frame = audio[6*44100 : 6*44100 + 1024]
#print ('Frames:', frame)

# Extract the pitch curve
# PitchMelodia takes the entire audio signal as input (no frame-wise processing is required)
pitch_extractor = PredominantPitchMelodia(frameSize=fS, 
                                            hopSize=hS, 
                                            harmonicWeight=0.9,
                                            magnitudeThreshold=40,
                                            minDuration=300,
                                            numberHarmonics=20,
                                            timeContinuity=50,             #the maximum allowed gap duration for a pitch contour
                                            voiceVibrato=True,
                                            voicingTolerance=0.8

                                            ) #These are samples!!! To write in audio we need to use frames
pitch_values, pitch_confidence = pitch_extractor(audio)
#print(pitch_values.size) # This is an array
# Pitch is estimated on frames. Compute frame time positions
pitch_times = numpy.linspace(0.0, len(audio)/sr, len(pitch_values) ) # Return seconds

 #Those two has the same size so the graph will be constructed by the times and the pitches
#pitch_times = [x for x in pitch_times]
#pitch_values = [x for x in pitch_values] 
#print(len(pitch_times))
#print((pitch_values))

# extract the indexes of the pitch times
prev_index=0
elements = []
for i in range(len(pitch_values)):
    num = pitch_values[i]
    if num > 0 and prev_index == 0:
        elements.append(i)
    prev_index = num

# Convert the indexes into samples
sample_segments = [n*hS for n in elements]
print(sample_segments)

numOfSegmetnts = len(sample_segments)

folder = 'Segments/'
if not os.path.exists(folder):
		os.makedirs(folder)

def audio_segments_generator(file):
    for cont in range(numOfSegmetnts - 1):
        posFrameInit = sample_segments[cont]
        posFrameEnd = sample_segments[cont + 1]
        MonoWriter(filename=folder + '{:03d}'.format(cont) + '_' + file,  format='wav', sampleRate=sr)(audio[posFrameInit:posFrameEnd])
    posFrameInit = sample_segments[numOfSegmetnts-1]
    # For last segment
    posFrameEnd = audio.size
    MonoWriter(filename=folder + '{:03d}'.format(cont) + '_' + file,  format='wav', sampleRate=sr)(audio[posFrameInit:posFrameEnd])

audio_segments_generator(file)

# Plot the estimated pitch contour and confidence over time

f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')
plt.show()

print('Amount of Segments:', numOfSegmetnts-1)

