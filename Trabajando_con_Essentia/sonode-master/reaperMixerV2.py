data = "C:\\Users\\hugos\\Dropbox\\UNAM\\ottoHugoCompartido\\Analyziador03_feb_2020\clases.txt"
file = open(data, "r")
dataLines = file.readlines()
path = "D:\\HUGO\\OttoAudio\\OttoSalida\\" #etcetcetc OJO QUE DEBE DE TERMINAR EN /

currentClassPos = {}
currentClassIndex = {}
numClass = -1

#obtiene los datos
for linea in dataLines:
  datosSeg = linea.strip().split("\t")
  clase = int(datosSeg[1])
  if clase > numClass: numClass = clase

numClass = numClass + 1 #porque 0 es una clase, 1 serian dos clases la 0 y la 1

#construye los tracks
for clase in range(numClass):
  RPR_InsertTrackAtIndex(clase, True)
  track = RPR_GetTrack(0,clase)
  RPR_SetTrackSelected(track, False)
  currentClassPos[clase] = 0 #todo empieza en contador cero
  currentClassIndex[clase] = 0

for linea in dataLines:
  datosSeg = linea.strip().split("\t")
  archivo = datosSeg[0]
  clase = int(datosSeg[1])
  track = RPR_GetTrack(0,clase) #0 es projecto activo
  RPR_SetTrackSelected(track, True)
  RPR_SetEditCurPos(currentClassPos[clase], False, False)
  RPR_InsertMedia(path + archivo + ".wav",0)
  itemId = RPR_GetTrackMediaItem(track, currentClassIndex[clase])
  pos = RPR_GetMediaItemInfo_Value(itemId, "D_POSITION")
  length = RPR_GetMediaItemInfo_Value(itemId, "D_LENGTH")
  currentClassPos[clase] = currentClassPos[clase] + (length-0.25)
  currentClassIndex[clase] = currentClassIndex[clase] + 1
  track = RPR_GetTrack(0,clase)
  RPR_SetTrackSelected(track, False)


