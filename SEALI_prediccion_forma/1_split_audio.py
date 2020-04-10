from pydub import AudioSegment
from pydub.utils import make_chunks
import glob
import csv
import os

# files = [
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/Eli_Keszler_-_01_-_Solo_-_Live_at_Cafe_OTO_Sat_25_May_2013-Amalgama.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/07 Heat C_W Moment - guitarra electrica.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/03-sink into, return - timbre plástico.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/CUERDAS DE NYLON + GONG 3 - Plastico.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/06-she looks as though she is.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/Adam_Scheflan_-_28_-_Improv4halasam - guitarra eléctrica.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/03_-_Strictly_Vertical.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/09 nakamura solo - electronico.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/10_akiyama_solo.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/13 Turntable With Guitar Amp - electronico.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/Ayelet_Lerman_-_30_-_Improvhalasam - timbre metálico opaco.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/04 Bowed.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/03 Scraped.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/01_-_The_Crow_Flew_After_Yi_Sang - metalico opaco.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/02 Beaten.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/impro con ishtar.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/anthony braxton - to composer john cage - airoso.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/03-intuición 2.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/01 Part I - electronico.wav",
# "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/impro_clasificacion/01 Mixt.wav"
# ]

files = ['/media/atsintli/4D12FE8D63A43529/Doctorado/github/SEALI_prediccion_forma/03 Scraped.wav',
#'/media/atsintli/4D12FE8D63A43529/Doctorado/github/SEALI_prediccion_forma/10_akiyama_solo.wav' 
] 

# detect the current working directory and print it
path = os.getcwd()
#print ("The current working directory is %s" % path)

# define the name of the directory to be created
path = path + "/segments/"

os.mkdir(path)

lenFiles = (len(files))
n = 0

while n < lenFiles:
    myaudio = AudioSegment.from_file(files[n], "wav")
    n = n+1
    chunk_length_ms = 2000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    lenChunks = (len(chunks))
    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{:02d}.wav".format(i)
        print ("exporting", chunk_name)
        chunk.export((path + "{:03d}".format(i+1) + "_" + str(lenChunks) + "_" + chunk_name), format="wav")

#Export all of the individual chunks as wav files

