(
~periodico = Array.new;
~folder = PathName.new("/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/periodico");
(
~folder.entries.do({
	arg path;
	~periodico = ~periodico.add(Buffer.read(s, path.fullPath));
});
);

~caotico = Array.new;
~folder = PathName.new("/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/caotico");
(
~folder.entries.do({
	arg path;
	~caotico = ~caotico.add(Buffer.read(s, path.fullPath));
});
);

~complejo = Array.new;
~folder = PathName.new("/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/complejo");
(
~folder.entries.do({
	arg path;
	~complejo = ~complejo.add(Buffer.read(s, path.fullPath));
});
);

~fijo = Array.new;
~folder = PathName.new("/media/atsintli/4D12FE8D63A43529/Doctorado/Data/Categorias/fijo");
(
~folder.entries.do({
	arg path;
	~fijo = ~fijo.add(Buffer.read(s, path.fullPath));
});
);
);

(
SynthDef(\play, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(2, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.5, 0.5, 0 ], [5, 2, 2]));
	sig = sig * env;
	Out.ar(out, sig);
		}).add;

SynthDef(\play1, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(2, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.5, 0.5, 0 ], [5, 4, 2]));
	sig = sig * env;
	Out.ar(out, sig);
		}).add;

SynthDef(\play2, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(2, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.5, 0.5, 0 ], [5, 10, 2]));
	sig = sig * env;
	Out.ar(out, sig);
		}).add;

SynthDef(\play3, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(2, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.5, 0.5, 0 ], [5, 4, 2]));
	sig = sig * env;
	Out.ar(out, sig);
		}).add;
)