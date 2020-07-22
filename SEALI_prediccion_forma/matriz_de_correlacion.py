#Matriz de correlación

import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys

audioname = ("example2.wav")

y, sr = librosa.load(audioname)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)

# Find nearest neighbors in MFCC space
R1 = librosa.segment.recurrence_matrix(mfcc)

# Or fix the number of nearest neighbors to 5
R2 = librosa.segment.recurrence_matrix(mfcc, k=5)

# Suppress neighbors within +- 7 samples
R3 = librosa.segment.recurrence_matrix(mfcc, width=7)

# Use cosine similarity instead of Euclidean distance
R4 = librosa.segment.recurrence_matrix(mfcc, metric='cosine')

# Require mutual nearest neighbors
R5 = librosa.segment.recurrence_matrix(mfcc, sym=True)

# Use an affinity matrix instead of binary connectivity
R6 = librosa.segment.recurrence_matrix(mfcc, mode='affinity')

# Plot the feature and recurrence matrices
plt.figure(figsize=(8, 4))
plt.subplot(2, 3, 1)
librosa.display.specshow(R1, x_axis='time', y_axis='time')
plt.title('Binary recurrence (symmetric)')

plt.subplot(2, 3, 2)
librosa.display.specshow(R2, x_axis='time', y_axis='time')
plt.title('Binary recurrence (symmetric)')

plt.subplot(2, 3, 3)
librosa.display.specshow(R3, x_axis='time', y_axis='time')
plt.title('Binary recurrence (symmetric) DIF')

plt.subplot(2, 3, 4)
librosa.display.specshow(R4, x_axis='time', y_axis='time')
plt.title('Binary recurrence (symmetric)')

plt.subplot(2, 3, 5)
librosa.display.specshow(R5, x_axis='time', y_axis='time')
plt.title('Binary recurrence (symmetric)')

plt.subplot(2, 3, 6)
librosa.display.specshow(R6, x_axis='time', y_axis='time', cmap='magma_r')
plt.title('Affinity recurrence')

plt.tight_layout()
plt.show()