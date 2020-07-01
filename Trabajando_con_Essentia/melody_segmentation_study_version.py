from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
import numpy as np
import essentia
import essentia.standard as es
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default
from essentia.standard import *
import numpy
import os
import glob
import sys

sys.setrecursionlimit(20000)
###############################Melody detection#################################

#Create folder for segments

folder = 'Segments/'
if not os.path.exists(folder):
		os.makedirs(folder)

sr = 44100
hS = 1024
fS=2048

def concat_audio (frame_count, segment_count, segment_list,audio_list, file_index):
    print("frame, segment", frame_count,segment_count, file_index)
    #print("len", len(segment_list), segment_count)
    if frame_count < len(segment_list[segment_count])-1:
        startFrame = segment_list[segment_count][frame_count]
        endFrame = segment_list[segment_count][frame_count + 1]
        filename = folder + '{:04d}'.format(file_index) + '_' + "segment.wav"
        #print(filename)
        MonoWriter(filename=filename,  format='wav', sampleRate=sr)(audio_list[segment_count][startFrame:endFrame])
        concat_audio(frame_count+1, segment_count, segment_list, audio_list,file_index+1)
    else:
        if segment_count < len(segment_list)-1:
            concat_audio(0, segment_count+1, segment_list, audio_list, file_index)

#Extract the pitch curve
#PitchMelodia takes the entire audio signal as input (no frame-wise processing is required)
def audio_segments_generator(audio):
    pitch_extractor = PredominantPitchMelodia(frameSize=fS, 
                                            hopSize=1024,
                                            filterIterations=3, 
                                            harmonicWeight=0.8,     #harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay)
                                            magnitudeThreshold=40, #    spectral peak magnitude threshold (maximum allowed difference from the highest peak in dBs)
                                            guessUnvoiced=False,
                                            minDuration=10, #    the minimum allowed contour duration [ms]
                                            numberHarmonics=20,
                                            timeContinuity=100,             #the maximum allowed gap duration for a pitch contour
                                            voiceVibrato=False,
                                            voicingTolerance=0.8,    #allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation)
                                            pitchContinuity=27.5625, #  27.5625  pitch continuity cue (maximum allowed pitch change during 1 ms time period) [cents]
                                            peakFrameThreshold=0.7 #0.45 funcionó para bach partita de violin per-frame salience threshold factor (fraction of the highest peak salience in a frame)
                                            ) #These are samples!!! To write in audio we need to use frames

    # Pitch is estimated on frames. Compute frame time positions
    pitch_values, pitch_confidence = pitch_extractor(audio)
    #pitch_times = numpy.linspace(0.0, len(audio)/sr, len(pitch_values) ) # Return seconds
    prev_index=0
    elements = []
    #Filter segments from data retrived
    for i in range(len(pitch_values)):
        num = pitch_values[i]
        if num > 0 and prev_index == 0:
            elements.append(i)
        prev_index = num
    # Convert the indexes into samples
    sample_segments = [n*hS for n in elements]
    audioSize = audio.size
    sample_segments.insert(0,0)
    sample_segments.append(audioSize)
    #print(len(elements))
    ##############################################################
    #Plot the estimated pitch contour and confidence over time

    # f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].plot(pitch_times, pitch_values)
    # axarr[0].set_title('estimated pitch [Hz]')
    # axarr[1].plot(pitch_times, pitch_confidence)
    # axarr[1].set_title('pitch confidence')
    # plt.show()
    print('Amount of Segments:', len(sample_segments))

    return sample_segments

allSampleList = []
allAudio = []
# Load audio files from folder

for audio_files in glob.glob( 'audio/' + "*.wav" ):
    audio = MonoLoader(filename=audio_files)()
    sampleSegments = audio_segments_generator(audio)
    allSampleList.append(sampleSegments)
    allAudio.append(audio)
#print("samples", allSampleList)

concat_audio(0,0,allSampleList,allAudio,0)