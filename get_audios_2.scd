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
SynthDef(\caotico, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(2, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.02, 0.02, 0 ], [2, 30, 2]));
	sig = sig * env;
	Out.ar(out, sig);
		}).add;

SynthDef(\complejo, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(2, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.2, 0.2, 0 ], [2, 20, 2]));
	sig = sig * env;
	Out.ar(out, sig);
		}).add;

SynthDef(\fijo, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env;
	sig = PlayBuf.ar(2, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.1, 0.1, 0 ], [5, 30, 2]));
	sig = sig * env;
	Out.ar(out, sig);
		}).add;

SynthDef(\periodico, {

	arg amp=1, out=0, buf, rate=1, loop = 0, start;
	var sig, env, rev;
	sig = PlayBuf.ar(2, buf, rate, 1, start, loop, doneAction: 2);
	env = EnvGen.kr(Env.new([0, 0.071, 0.071, 0 ], [2, 20, 2]));
	sig = sig * env;
	Out.ar(out, GVerb.ar(sig, 30, 0.01));
		}).add;

SynthDef(\help_Klank, { arg out=0, i_freq, gate = 1;
    var klank, n, harm, amp, ring, env;
    // harmonics
    harm = \harm.ir(Array.series(4, 1, 1).postln);
    // amplitudes
    amp = \amp.ir(Array.fill(4, 0.05));
    // ring times
    ring = \ring.ir(Array.fill(4, 1));
    klank = Klank.ar(`[harm, amp, ring], {ClipNoise.ar(0.0003)}.dup, i_freq);
	env = EnvGen.kr(Env.new([0, 2, 0.9, 0 ], [5, 13, 2]), gate, doneAction:  Done.freeSelf);
    Out.ar(out, env*klank);
}).add;
)