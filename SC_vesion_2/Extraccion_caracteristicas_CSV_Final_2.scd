(
~dir = thisProcess.nowExecutingPath.dirname; //genera variable global del directorio donde estamos, va a dar error si el archivo no esta salvado.

~getFilePathsOn = {|pathToFolder|
    PathName(pathToFolder).entries.collect(_.fullPath)
};
)

(
var filePath = ~dir ++ "/../dataBaseAsMatrix_scmir_21277.csv"; //este es el archivo que se va a crear con nuestros datos
//Diccionario que contiene la base de datos de ejemplos de audio para analizar

//var inputData = (
//	"": ~getFilePathsOn.(~dir++"/../Clase_6"),

var inputData = (
	"0": ~getFilePathsOn.("/media/atsintli/Samsung/audioClases/Clase_0/"),
	"1": ~getFilePathsOn.("/media/atsintli/Samsung/audioClases/Clase_1/"),
	"2": ~getFilePathsOn.("/media/atsintli/Samsung/audioClases/Clase_3/"),
	"3": ~getFilePathsOn.("/media/atsintli/Samsung/audioClases/Clase_3/"),
	"4": ~getFilePathsOn.("/media/atsintli/Samsung/audioClases/Clase_4/"),
	"5": ~getFilePathsOn.("/media/atsintli/Samsung/audioClases/Clase_5/"),
	"6": ~getFilePathsOn.("/media/atsintli/Samsung/audioClases/Clase_6/"),
);

var classNames = inputData.keys.asArray.sort; // obtenemos los nombres de las clases de acuerdo al diccionario en orden alfabético

var sources = classNames.collect({|archives| inputData[archives]}).postln; //obtenemos las rutas de los archivos de audio en orden
//var sources = classNames.collect({inputData[_]}); //otra forma de escribir lo mismo

var ventaneo = 0; //Indice de ventaneo

//var features = [[MFCC,4],[BeatStatistics]];
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness]];
//var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80]];

//var features = [[MFCC,13],[Loudness]];
//var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],[BeatStatistics],[MFCC]];

var file;
file = File(filePath, "w");
file.write(~stringify.(~getAudioFeatures.(sources, classNames, features, ~ventanizar, ventaneo)));
file.close;
)

(
var filePath = ~dir ++ "/testset_82.csv"; //este es el archivo que se va a crear con nuestros datos

//Diccionario que contiene la base de datos de ejemplos de audio para analizar

var inputData = (
	//"caotico": ~getFilePathsOn.(~dir++"/caotico-test"),
	"complejo": ~getFilePathsOn.(~dir++"/complejo-test"),
	"fijo": ~getFilePathsOn.(~dir++"/fijo-test"),
	"periodico": ~getFilePathsOn.(~dir++"/periodico-test"),
);

var classNames = inputData.keys.asArray.sort; // obtenemos los nombres de las clases de acuerdo al diccionario en orden alfabético

var sources = classNames.collect({|archives| inputData[archives]}).postln; //obtenemos las rutas de los archivos de audio en orden
//var sources = classNames.collect({inputData[_]}); //otra forma de escribir lo mismo

var ventaneo = 82; //Indice de ventaneo

//var features = [[SpectralEntropy],[Chromagram],[SpecCentroid],[BeatStatistics]];
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],[BeatStatistics]];
var file;

file = File(filePath, "w");
file.write(~stringify.(~getAudioFeatures.(sources, classNames, features, ~ventanizar, ventaneo)));
file.close;
)