(
SynthDef(\micro, {
	Out.ar(8, SoundIn.ar([0,1]))
}).add;

// this will record to the disk
SynthDef("help-Diskout", { |bufnum|
	DiskOut.ar(bufnum, In.ar(8, 2));
}).add;

~recordSomething = {|filePath, fileNum, bufferDur, cb, i|
	Task({
		var x = Synth(\micro);
		var b, d;

		0.02.wait;

		// allocate a disk i/o buffer
		b= Buffer.alloc(s, 2.pow(15), 2);

		// create an output file for this buffer, leave it open
		b.write(filePath, "wav", "int16", 0, 0, true);
		// create the diskout node; making sure it comes after the source
		d = Synth.tail(nil, "help-Diskout", ["bufnum", b]);
		("Starting recording #" + i).postln;
		(bufferDur).wait;
		// stop recording
		d.free;
		// stop the bubbles
		x.free;
		// close the buffer and the soundfile
		b.close;
		// free the buffer
		b.free;

		("Finished recording #" + i).postln;
		cb.(filePath);
	}).play;
};

~startRec = {|dir, bufferDur, cb|
	var task = Task({
		inf.do({|i|
			var fileNum = i % 100;
			var filePath = (dir++"/"++fileNum++".wav").standardizePath;
			~recordSomething.(filePath, fileNum, bufferDur, cb, i);
			(bufferDur+0.02).wait;
		});
	});
	task.play;
};
);

// descomentar y correr individualmente
/*
(
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],
	[BeatStatistics]];
var ventaneo = 20;
t = ~startRec.("~/Desktop/recs", 0.5, {|fileName|
	Task({
		0.02.wait;
		~res = ~getAudioFeatures.([[fileName]], nil, features, ~ventanizar, ventaneo);
		~data = ~res[\unknown].flatten.flatten;
	}).play
});
)

t.stop

