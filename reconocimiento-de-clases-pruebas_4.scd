~dir = thisProcess.nowExecutingPath.dirname;

(
var featureinfo = [[Chromagram]];

var filePath = ~dir ++ "/data.arff";

//array of arrays, one array for each genre class/category

var inputData = (
	"fijo": [~dir ++ "/01_fijo_Fragmente_Stille_An_Diotima_05.wav"],
	"periodico": [~dir ++ "/02_Periodico_Cowell_Pulse_06.wav"],
	"caotico": [~dir ++ "/03_caotico_iancu_dumitrescu_nucleons_chaotiques_transe_03.wav"],
	"complejo":[~dir ++ "/04_complejo_Section_24_01.wav"],
);

var classNames = inputData.keys.asArray;

var arff = SCMIR.createARFF(filePath, 12, classNames);

var sources = classNames.collect(inputData[_]);

var stringify = {|smirs|
    smirs.keys.collect({|className|
        var data = smirs.at(className).flatten;
        data.collect({|ventana| ventana.asList.add(className).join(",")})
    }).asArray.flatten.join("\n");
};
var ventanizar = ventanizar = {|len, arr|
    var arrOverLen = arr.size/len;
    var shape = (1..len)!(arrOverLen.asInteger)
    // ++[(1!(arr.size - (arrOverLen.asInteger*len)))]; // agrega una ventana parcial con sun size menor a len
    ;
    var ventanas = arr.reshapeLike(shape);
    ventanas;
};


var file;

{
    var smirs = sources.inject(
        Dictionary.new,
        {|dict, array, i|

            var data = array.collect{|filename|
                var file;

                file = SCMIRAudioFile(filename,featureinfo);

                file.extractFeatures();

                file.featuredata.asArray;
            };
			dict.put(classNames[i], data.collect(ventanizar.(12, _)));
            dict;

    });
	arff.close;
	file = File(filePath, "a+");
	file.write(stringify.(smirs));
	file.close;
}.fork;
)


// Test de función previa para ventanizar
// (
// var len = 43;
// var arr = (0..200);// array de 0 a 200
// var arrOverLen = arr.size/len;
// var shape = (1..len)!(arrOverLen.asInteger)
// // ++[(1!(arr.size - (arrOverLen.asInteger*len)))];
// ;
// var ventanas = arr.reshapeLike(shape);
//
// //tests
// ventanas.debug("ventanas");
// arr.size.debug("arr size del original");
// ventanas.collect(_.size).debug("tamaño de ventanas").sum.debug("arr size nuevo");
// arr.last.debug("ultimo número original");
// ventanas.last.last.debug("ultimo numero");
// )


(
var inputData = (
        "fijo": [
        "/media/atsintli/4D12FE8D63A43529/Doctorado/audios/Categorias/Fijo/01_fijo_Fragmente_Stille_An_Diotima_16.wav",
    ],
    "periodico": [
        "/media/atsintli/4D12FE8D63A43529/Doctorado/audios/Categorias/Periodico/02_periodico_Fragmente_Stille_An_Diotima_25.wav"
    ],
);
var classNames = inputData.keys.asArray;
a = classNames.collect(inputData[_]);
classNames;
)