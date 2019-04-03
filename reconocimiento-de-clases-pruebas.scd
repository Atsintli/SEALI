~dir = thisProcess.nowExecutingPath.dirname;

(
var featureinfo = [[Chromagram, 12], [MFCC, 13]];

var numFrames = featureinfo.collect({|descriptorArr| descriptorArr[1]}).sum;

var filePath = ~dir ++ "/data.arff";

//array of arrays, one array for each genre class/category

var inputData = (
	"fijo": [~dir ++ "/01_fijo_Fragmente_Stille_An_Diotima_05.wav"],
	"periodico": [~dir ++ "/02_Periodico_Cowell_Pulse_06.wav"],
	"caotico": [~dir ++ "/03_caotico_iancu_dumitrescu_nucleons_chaotiques_transe_03.wav"],
	"complejo":[~dir ++ "/04_complejo_iancu_dumitrescu_nucleons_chaotiques_transe_06.wav"],
);

var classNames = inputData.keys.asArray.sort;

var sources = classNames.collect({|className| inputData[className]});
// var sources = classNames.collect(inputData[_]);

var stringify = {|scmirs|
    scmirs.keys.asArray.sort.collect({|className|
        var data = scmirs.at(className).flatten;
        data.collect({|ventana|
            ventana.asList.add(className).join(",")
        })
    })
    .flatten.join("\n");
};

var ventanizar = ventanizar = {|len, arr|
    var totalFrames = (arr.size/len).asInteger; // si totalFrames < 1 entonces se ventanizar devuelve un array vacío
    var shape = (1..len)!totalFrames;
    // ++[(1!(arr.size - (totalFrames.asInteger*len)))]; // agrega una ventana parcial con un size menor a len

    var ventanas = arr.reshapeLike(shape);
    ventanas;
};

    var scmirs = sources.inject(
        Dictionary.new,
        {|dict, array, index|

            var data = array.collect{|filename|
                var file = SCMIRAudioFile(filename, featureinfo);
                var data;
                file.extractFeatures();

                data = file.featuredata.asArray;
                file.numframes.debug("numframes");
                data.size.debug("size");
                data;
            };
            dict.put(classNames[index].debug("className"), data.collect(ventanizar.(numFrames*43, _)));
            dict;

    });

    var arff = SCMIR.createARFF(filePath, numFrames, classNames);
    var file;
	arff.close;
	file = File(filePath, "a+");
	file.write(stringify.(scmirs));
	file.close;
)


