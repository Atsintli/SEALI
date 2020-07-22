(
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
    scmirs;
};
)
