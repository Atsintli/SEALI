(
// función para configurar los datos extraídos de scmirs.
var stringify = {|scmirs|
    scmirs.keys.asArray.sort.collect({|className|
        var data = scmirs.at(className).flatten;
        data.collect({|ventana|
            if(className != \unknown,
                {ventana.asList.add(className).join(",")},
                {ventana.asList.join(",")}
            )

        })
    })
    .flatten.join("\n");
};

//función para extraer las características de los audios
~getAudioFeatures = {|sources, classNames = ([\unknown]), features, ventanizar, ventaneo|
    var scmirs = sources.inject(
        Dictionary.new,
        {|dict, array, index|
            var numFeatures;
            var dataArr = array.collect{|filename|
                var data;
                var file = SCMIRAudioFile(filename, features);
                numFeatures = file.numfeatures;
                file.extractFeatures();
                data = file.featuredata.asArray;
                data;
            };
            dict.put(
                classNames[index].debug("className"),
                dataArr.collect({|data| ventanizar.(numFeatures*ventaneo, data)}));
            dict;
    });
    stringify.(scmirs, classNames); // features
};
)
