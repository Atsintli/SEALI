~dir = thisProcess.nowExecutingPath.dirname; //genera variable global del directorio donde estamos, va a dar error si el archivo no esta salvado.

(
~getFilePathsOn = {|pathToFolder|
    PathName(pathToFolder).entries.collect(_.fullPath)
};

// ~getFilePathsOn.(~dir++"/caoticos")
)

(
var filePath = ~dir ++ "/data.arff"; //este es el archivo que se va a crear con nuestros datos

//Diccionario que contiene la base de datos de ejemplos de audio para analizar

var inputData = (
	"caotico": ~getFilePathsOn.(~dir++"/caotico"),
	"complejo": ~getFilePathsOn.(~dir++"/complejo"),
	"fijo": ~getFilePathsOn.(~dir++"/fijo"),
	"periodico": ~getFilePathsOn.(~dir++"/periodico"),
);

var classNames = inputData.keys.asArray.sort; // obtenemos los nombres de las clases de acuerdo al diccionario en orden alfabético

var sources = classNames.collect({|archives| inputData[archives]}).postln; //obtenemos las rutas de los archivos de audio en orden
//var sources = classNames.collect({inputData[_]}); //otra forma de escribir lo mismo

var ventaneo = 10; //Indice de ventaneo

var features = [[MFCC, 2],[Chromagram],[SpecCentroid],[SpecPcile, 0.95],[SpecPcile, 0.8],[SpecFlatness],[FFTCrest], [FFTSpread],[FFTSlope],[SensoryDissonance],[Onsets,\rcomplex]];

var file;

file = File(filePath, "w");
file.write(~getAudioFeatures.(sources, classNames, features, ~ventanizar, ventaneo));
file.close;
)



