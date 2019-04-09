~dir = thisProcess.nowExecutingPath.dirname; //genera variable global del directorio donde estamos, va a dar error si el archivo no esta salvado.

(
var filePath = ~dir ++ "/data.arff"; //este es el archivo que se va a crear con nuestros datos

var inputData = (
	"caotico": [~dir ++ "/03_caotico_iancu_dumitrescu_nucleons_chaotiques_transe_03.wav"],
	"complejo": [~dir ++ "/04_complejo_iancu_dumitrescu_nucleons_chaotiques_transe_06.wav"],
	"fijo": [~dir ++ "/01_fijo_Fragmente_Stille_An_Diotima_05.wav"],
	"periodico": [~dir ++ "/02_Periodico_Cowell_Pulse_06.wav"],
); //Diccionario que contiene la base de datos de ejemplos de audio para analizar


var classNames = inputData.keys.asArray.sort; // obtenemos los nombres de las clases de acuerdo al diccionario en orden alfabético

var sources = classNames.collect({|archives| inputData[archives]}); //obtenemos las rutas de los archivos de audio en orden
//var sources = classNames.collect({inputData[_]}); //otra forma de escribir lo mismo

var ventaneo = 10; //Indice de ventaneo

//función para determinar el ventaneo
var ventanizar = {|len, arr|
	var totalFrames = (arr.size/len).asInteger;
	var shape = (1..len)!totalFrames;
	var ventanas = arr.reshapeLike(shape);
	ventanas;
};

var features = [[MFCC, 2],[Chromagram],[SpecCentroid],[SpecPcile, 0.95],[SpecPcile, 0.8],[SpecFlatness],[FFTCrest], [FFTSpread],[FFTSlope],[SensoryDissonance],[Onsets,\rcomplex]];

//contar componentes de los descriptores.
//var numFeatures = features.collect({|descriptorArr| descriptorArr[1]}).sum;//si mando numFeatures que cuenta el segundo valor del array features y el descriptor no indica cuatas características tiene, numFeatures no funciona, esto es obvio porque le estoy diciendo que sume el segundo valor de cada array.


//función para extraer las características de los audios
var scmirs = sources.inject(
	Dictionary.new,
	{|dict, array, index|

		var data = array.collect{|filename|
			var file = SCMIRAudioFile(filename, features);
			file.extractFeatures();
			~numFeatures = file.numfeatures;
			data = file.featuredata.asArray;
			file.numfeatures.debug("numfeatures");
            data.size.debug("size");
			data;
		};
		dict.put(classNames[index].debug("className"), data.collect(ventanizar.(~numFeatures*ventaneo, _)));
		dict;
	});

// función para configurar los datos extraídos de scmirs.
var stringify = {|scmirs|
	scmirs.keys.asArray.sort.collect({|classNames|
		var data = scmirs.at(classNames).flatten;
		data.collect({|ventana|
			ventana.asList.add(classNames).join(",")
		})
	})
	.flatten.join("\n");
};

//crear arff y file
var arff = SCMIR.createARFF(filePath, ~numFeatures*ventaneo, classNames);
var file;
arff.close;
file = File(filePath, "a+");
file.write(stringify.(scmirs));
file.close;
)
