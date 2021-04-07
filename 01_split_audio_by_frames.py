from pydub import AudioSegment
import math

cut_size = 0.1

class SplitWavAudioMubin():
    
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * cut_size * 1000
        t2 = to_min * cut_size * 1000
        split_audio = self.audio[t1:t2]
        #split_audio = 0.01.fade_in(2).fade_out(2)
        split_audio.export(self.folder + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / cut_size)
        for i in range(0, total_mins, min_per_split):
            split_fn = 'splits/' + "{:06d}".format(i) + ".wav"
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')

folder = 'audiotest'
file = 'sinkintoreturn.wav'
split_wav = SplitWavAudioMubin(folder, file)
split_wav.multiple_split(min_per_split=1)