// función para agregar el nombre de la clase.
~stringify = {|scmirs|
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