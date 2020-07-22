import librosa
import sys
import os
import glob
import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf
from pylab import *


#fileName = sys.argv[1]
fileName = "02_periodico_Taira_Hierophonie.wav"
fileJustName = fileName.split('.')[-2]
if not os.path.exists(fileJustName):
		os.makedirs(fileJustName)

print (fileJustName)

y, sr = librosa.load(fileName)
#onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=False)
#print (onset_frames)
onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                        hop_length=512,
                                        center = True,
                                        n_mels=256,
                                         )
#print (onset_env)
#onset_bt = librosa.onset.onset_backtrack(onset_raw, onset_env)
#print (onset_raw)
peaks = librosa.util.peak_pick(onset_env, 7, 7, 7, 7, 0.95, 20)
#print (peaks)
times = librosa.frames_to_samples(peaks)
#print (times)

file = open(fileJustName + '_durs.txt', 'w')
amountOfSegments = times.size
for cont in range(amountOfSegments - 1):
	posFrameInit = times[cont]
	posFrameEnd = times[cont + 1]
	duracionSeg = posFrameEnd - posFrameInit
	#print("duracion de segmento = " + str(duracionSeg))
	file.write(str(duracionSeg) + "\n")
	librosa.output.write_wav(fileJustName + '/' + fileJustName + '_' 
		+ '{:02d}'.format(cont) + ".wav", y[posFrameInit:posFrameEnd-6625], sr)
file.close()
posFrameInit = times[amountOfSegments-1]
posFrameEnd = y.size
librosa.output.write_wav(fileJustName + '/' + fileJustName + '_' 
	+ '{:02d}'.format(amountOfSegments-1) + ".wav", y[posFrameInit:posFrameEnd-6625], sr) #-6625

times = librosa.samples_to_time(times)
#print times

amountOfSegments = times.size

file = open(fileJustName + '_times.txt', 'w')
for cont in range(amountOfSegments -1):
	posFrameInit = times[cont]
	file.write(str(posFrameInit) + "\n")
file.close()


# Plot
timess = librosa.frames_to_time(np.arange(len(onset_env)),
                               sr=sr, hop_length=512)
plt.figure()
ax = plt.subplot(2, 1, 2)
D = librosa.stft(y)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.subplot(2, 1, 1, sharex=ax)
plt.plot(timess, onset_env, alpha=0.8, label='Onset strength')
plt.vlines(timess[peaks], 0,
           onset_env.max(), color='r', alpha=0.8,
           label='Selected peaks')
plt.legend(frameon=True, framealpha=0.8)
plt.axis('tight')
plt.tight_layout()
plt.show()

print ("numero de segmentos = " + str(amountOfSegments))
print ('Termine de Cortar')