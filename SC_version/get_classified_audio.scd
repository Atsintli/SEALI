~folder = "/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/complejo/";

(
Task({
	100.do({|i|
		FSSound.textSearch(query: "complex", filter: "type:wav",
			params:('page': i+1), action:{|pager|
					~snd = pager[1];
					~snd.name.postln;
				}
			);
		"waiting for data to retrieve!".postln;
		0.5.wait;
		~snd.retrieve(~folder, {
			//~buf = Buffer.read(s, ~folder ++ ~snd.name);
			(~folder ++ ~snd.name).postln;
			"done!".postln;
		});
	});
}).play
)