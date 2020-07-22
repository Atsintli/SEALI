import sys
import os
import subprocess

#in_dir = 'audioClases/'
#out_dir = 'audio_descriptors_anotated_data/'

in_dir = 'Segments/'
out_dir = 'audio_descriptors_extended/'

print(out_dir)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def extractor(root,fileName):
	print("esta analizando " + fileName)
	file_name, file_extension = os.path.splitext(fileName)
	subprocess.call(["./streaming_extractor_music",
     root + "/"+ fileName, out_dir + "/"+ file_name + ".json"])


for root, dirs, files in os.walk(in_dir):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        #print(len(path) * '---', file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension.lower() == ".wav":
        	print(len(path) * '---', file)
        	extractor(root,file)