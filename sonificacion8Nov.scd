/////////////////////////////////////////////////////////////////////////////////
h = Buffer.alloc(s, 8192, 1);
Ndef(\sin).play;
Ndef(\sin).fadeTime=5;
Ndef(\buf).play;
Ndef(\buf).fadeTime=5;

Ndef(\sin).free

//modo propositor

(
OSCdef (\osc2, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	//caóticos

	if((caotico < 0.0002),{
		Ndef(\sin, {
			Silent.ar
		})
	}
	);

	if((caotico > 0.0003) && (caotico < 0.02),{
		Ndef(\buf, {
			var sig, env;
			sig = PlayBuf.ar(2, ~fijo[0..~fijo.size].choose, (-8..-1).choose.midiratio, 1, doneAction: 2);
			env = EnvGen.kr(Env.new([0, 0.5, 0.8, 0], [2, 30, 2]));
			sig = sig * env;
		})
	}
	);

	if(
		(caotico > 0.03) && (caotico < 0.09), //condition
		{Ndef(\sin,{
			Resonz.ar(
				Mix.fill(28,{
					var freq, numcps;
					numcps= rrand(2,20);
					Pan2.ar(Gendy4.ar(1, 100, 1.0.rand,1.0.rand,233,235, 1, 1.0.rand, numcps, 12, 1/(28.sqrt)), 1.0.rand2)}),MouseX.kr(233,466), MouseY.kr(0.1,4.0))
		})}
	);

	if((caotico > 0.1) && (caotico < 0.2),
		{
			Ndef(\sin, {
				var in, freq, hasfreq;
				in = SoundIn.ar(0);
				#freq, hasfreq = Tartini.kr(in);
				SinOsc.ar((freq*1.1).lag(0.2)!2,0,0.2)})
		}
	);

	if((caotico > 0.3),
		{
			Ndef (\sin, {|amp|
				//var freqs = [115, 105];
				Mix.new(Pan2.ar(SinOsc.ar([
					periodico.linexp(0.1**17, 1, 222, 6002),
					complejo.linexp(0.1**17, 1, 309, 1009),
					fijo.linexp(0.1**17, 1, 305, 380),
					periodico.linexp(0.1**17, 1, 513, 6000),
				], 0, 0.1), complejo.linexp(0.1**17, 0.9, -0.5, 0.5), 0.2));
			})
		}
	);

	//complejos

	if((complejo > 0.9999),{
		Ndef(\buf, {
			var sig, env;
			sig = PlayBuf.ar(2, ~complejo[0..~complejo.size].choose, (-7..-1).choose.midiratio, 1, doneAction: 2);
			env = EnvGen.kr(Env.new([0, 0.6, 0.5, 0], [2, 25, 2]));
			sig = sig * env;
	})	}
	);

	if((complejo > 0.01) && (complejo < 0.5),{
		Ndef(\buf, {
			var sig, env;
			sig = PlayBuf.ar(2, ~fijo[0..~fijo.size].choose, (-8..-1).choose.midiratio, 1, doneAction: 2);
			env = EnvGen.kr(Env.new([0, 0.3, 0.3, 0], [2, 25, 2]));
			sig = sig * env;
		})
	}
	);

	//Fijos

	if((fijo > 0.3) && (fijo < 0.6),{
		Synth(\caotico, [\buf, ~caotico[0..~caotico.size].choose, \rate, (-5..5).choose.midiratio])
	}
	);

	if((fijo > 0) && (fijo < 0.9),{
		Ndef(\sin,
			{
				|norm = 0.2, dryrevlev= 0.05, controlfreq, room = 20, newfreq|
				var in, freq, hasfreq, fft, amp, loudness, noise, brown_noise;
				var in_Gen, rev, env, onset, sig, ctrspec;

				in = SoundIn.ar(0);
				amp = Amplitude.kr(in, 0.01, 0.01);

				#freq, hasfreq = Tartini.kr(in, 20);
				fft = FFT(h, in);
				loudness = Loudness.kr(fft);
				onset = Onsets.kr(fft, 0.01);
				newfreq = freq.lag(10);

				brown_noise = WhiteNoise.ar([0.5,0.5], amp*0.8);
				env = EnvGen.kr(Env.new([0, 0.7, 0.5, 0 ], [0.5, 0.5, 1.1], 1, 2, nil), onset);
				noise = brown_noise * env;
				sig = Resonz.ar(noise, newfreq*0.99, 0.0001);
				//noise = HPF.ar(noise, 440);
				//Pan2.ar(in*noise, LFSaw.kr(loudness.linlin(20, 45, 2, 0.01)), 1);
				rev = Mix.ar(GVerb.ar(sig, room, 5, 0.8, 0.8, 10, 1));
				Normalizer.ar(DelayN.ar(rev, 10, 2.5), norm, 0.01);
	})}
	);

	//Periodicos

	if((periodico > 0.8) && (periodico < 1),{
		Synth(\periodico, [\buf, ~periodico[1..~periodico.size].choose, \rate, (-6..-5).choose.midiratio])
	}
	);

},'/clase',recvPort: 5006);
)


/////////////////////////////////////////////////////////////////////////////////
//modo Imitativo/acompañamiento/relax

(
OSCdef (\osc2, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	//caóticos

	if((caotico < 0.0002),{
		Ndef(\sin, {
			Silent.ar
		})
	}
	);

	if(
		(caotico > 0.03) && (caotico < 0.09), //condition
		{Ndef(\sin,{
			Resonz.ar(
				Mix.fill(28,{
					var freq, numcps;
					numcps= rrand(2,20);
					Pan2.ar(Gendy4.ar(1, 100, 1.0.rand,1.0.rand,233,235, 1, 1.0.rand, numcps, 12, 1/(28.sqrt)), 1.0.rand2)}),MouseX.kr(233,466), MouseY.kr(0.1,4.0))
		})}
	);

	if((caotico > 0.1) && (caotico < 0.2),
		{
			Ndef(\sin, {
				var in, freq, hasfreq;
				in = SoundIn.ar(0);
				#freq, hasfreq = Tartini.kr(in);
				SinOsc.ar((freq*2.1).lag(0.002)!2,0,0.8)})
		}
	);

	if((caotico > 0.3),
		{
			Ndef (\sin, {|amp|
				//var freqs = [115, 105];
				Mix.new(Pan2.ar(SinOsc.ar([
					periodico.linexp(0.1**17, 1, 222, 6002),
					complejo.linexp(0.1**17, 1, 309, 1009),
					fijo.linexp(0.1**17, 1, 305, 380),
					periodico.linexp(0.1**17, 1, 513, 6000),
				], 0, 0.1), complejo.linexp(0.1**17, 0.9, -0.5, 0.5), 0.2));
			})
		}
	);

	//complejos

	if((complejo > 0.9999),
		{
			Ndef(\sin, {
				var in, freq, hasfreq;
				in = SoundIn.ar(0);
				#freq, hasfreq = Tartini.kr(in);
				SinOsc.ar((freq*(0.1)).lag(0.02)!2,0,0.2)})
		}
	);

	if((complejo > 0.01) && (complejo < 0.5),{
		Ndef(\buf, {
			var sig, env;
			sig = PlayBuf.ar(2, ~complejo[0..~complejo.size].choose, (-8..-1).choose.midiratio, 1, doneAction: 2);
			env = EnvGen.kr(Env.new([0, 0.5, 0.5, 0], [2, 25, 2]));
			sig = sig * env;
		})
	}
	);

	//Fijos

	if((fijo > 0.6) && (fijo < 1),{
		Synth(\fijo, [\buf, ~fijo[0..~fijo.size].choose, \rate, (-5..5).choose.midiratio])
	}
	);

	if((fijo > 0.000001) && (fijo < 0.5),{
		Ndef(\sin,
			{
				|norm = 0.2, dryrevlev= 0.05, controlfreq, room = 20, newfreq|
				var in, freq, hasfreq, fft, amp, loudness, noise, brown_noise;
				var in_Gen, rev, env, onset, sig, ctrspec;

				in = SoundIn.ar(0);
				amp = Amplitude.kr(in, 0.01, 0.01);

				#freq, hasfreq = Tartini.kr(in, 20);
				fft = FFT(h, in);
				loudness = Loudness.kr(fft);
				onset = Onsets.kr(fft, 0.01);
				newfreq = freq.lag(5);

				brown_noise = WhiteNoise.ar([0.5,0.5], amp*0.8);
				env = EnvGen.kr(Env.new([0, 0.7, 0.5, 0 ], [0.5, 0.5, 1.1], 1, 2, nil), onset);
				noise = brown_noise * env;
				sig = Resonz.ar(noise, newfreq*0.99, 0.01);
				//noise = HPF.ar(noise, 440);
				//Pan2.ar(in*noise, LFSaw.kr(loudness.linlin(20, 45, 2, 0.01)), 1);
				rev = Mix.ar(GVerb.ar(sig, room, 5, 0.8, 0.8, 10, 1));
				Normalizer.ar(DelayN.ar(rev, 10, 2.5), norm, 0.01);
	})}
	);

	//Periodicos

	if((periodico > 0.8) && (periodico < 1),{
		Ndef (\sin, {
			//var freqs = [115, 105];
			Mix.new(Pan2.ar(
				Resonz.ar(PinkNoise.ar([
					caotico.linlin(0.1**17, 1, 1, 2),
					fijo.linlin(0.1**17, 1, 1, 2),
				], 10),
				complejo.linlin(0.1**11, 0.1, 220, 15000),
				periodico.linlin(0.1**17, 0.7, 0.001, 0.015),
				),
				complejo.linlin(0.1**17, 0.9, -0.5, 0.5), 15));
		});
	}
	);

},'/clase',recvPort: 5006);
)

/////////////////////////////////////////////////////////////////////////////////

//modo ruptura

(
OSCdef (\osc2, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	if(caotico > 0.99){
		Synth(\fijo, [\buf, ~fijo[0..~fijo.size].choose, \rate, (-5..5).choose.midiratio])
	};

	if((caotico > 0.20) && (caotico < 0.80)){
		Synth(\fijo, [\buf, ~fijo[0..~fijo.size].choose, \rate, (-8..0).choose.midiratio]);
	};

	if(complejo > 0.9){
		Synth(\periodico, [\buf, ~periodico[0..~periodico.size].choose, \rate, (-5..5).choose.midiratio])

	};

	if(fijo > 0.9999999){

		Synth(\caotico, [\buf, ~caotico[0..~caotico.size].choose, \rate, (-5..5).choose.midiratio])

	};

	if(periodico > 0.9){

		Synth(\fijo, [\buf, ~fijo[1..~fijo.size].choose, \rate, (-5..5).choose.midiratio])

	};

},'/clase',recvPort: 5006);
)

////////////////////////////////////////////////////////////////////
//modo sinusoidal superpuesto ojo osc1 arriba osc2

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	//msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	(
		Ndef (\sin, {|amp|
			//var freqs = [115, 105];
			Mix.new(Pan2.ar(SinOsc.ar([
				caotico.linexp(0.1**17, 1, 222, 6013),
				complejo.linexp(0.1**17, 1, 309, 1009),
				fijo.linexp(0.1**17, 1, 305, 380),
				periodico.linexp(0.1**17, 1, 513, 6000),
			], 0, 0.8), complejo.linexp(0.1**17, 0.9, -0.5, 0.5), 0.2));
		});
	)

},'/clase',recvPort: 5006); // once only
)

(
OSCdef (\osc3, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	(

		Ndef (\sin, {
			//var freqs = [115, 105];
			Mix.new(Pan2.ar(
				Resonz.ar(PinkNoise.ar([
				caotico.linlin(0.1**17, 1, 1, 2),
				fijo.linlin(0.1**17, 1, 1, 2),
			], 10),
				complejo.linlin(0.1**11, 0.1, 220, 15000),
				periodico.linlin(0.1**17, 0.7, 0.001, 0.015),
				),
				complejo.linlin(0.1**17, 0.9, -0.5, 0.5), 15));
		});
	)

},'/clase',recvPort: 5006);
)
