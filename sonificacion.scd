(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

			// segment the data: split into different parameters
			c = case
				{caotico > 0.7} {
					//"send to param 1: ".post; i.postln;
					Ndef(\sin).fadeTime = 1;
					Ndef (\sin, {SinOsc.ar(MouseX.kr([120,121], [18000,18100], 5, 1,0,0))});
				}

				{complejo > 0.7} {
					//"send to param 2: ".post; i.postln;
					Ndef (\sin, {SinOsc.ar(XLine.kr([18000,18100], [120,121], 5, 1,0,0))});
				}

				{fijo > 0.7} {
					//"send to param 3: ".post; i.postln;
					Ndef (\sin, {

						SinOsc.ar(440)}

					);
			}
				{periodico > 0.7} {
					//"send to param 3: ".post; i.postln;
					Ndef (\sin, {SinOsc.ar(XLine.kr([200,2000], [120,121], 8, 1,0,0))}

					);
};

~dataMap.value;

},'/clase',recvPort: 5006); // once only
)

Ndef(\sin).play;
Ndef(\sin).fadeTime = 5

////////////////////////////////////////////////////////////////////////////////////////////

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	//msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	(
		// segment the data: split into different parameters
		c = case
		{caotico > 0.7} {
			//"send to param 1: ".post; i.postln;
			Ndef (\sin, {SinOsc.ar(MouseX.kr([120,121], [18000,18100], 3, 1,0,0),0,0.8)});
		}

		{complejo > 0.7} {
			//"send to param 2: ".post; i.postln;
			Ndef (\sin, {SinOsc.ar([16000,16100], 0, 0.8)});
		}

		{fijo > 0.7} {
			//"send to param 3: ".post; i.postln;
			Ndef (\sin, {

				SinOsc.ar([1000, 1001], [0,0.5], complejo.linlin(0.1**11, 1, 0.05, 0.8))}

			);
		}

		{periodico > 0.9} {
			//"send to param 3: ".post; i.postln;
			Ndef (\sin, {SinOsc.ar(XLine.kr([200,201], [245,240], 0.5, 1,0,0))}

			);
		};
	)

},'/clase',recvPort: 5006); // once only
)

Ndef(\sin).play;
Ndef(\sin).fadeTime = 5;

////////////////////////////////////////////////////////////////////////////////

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

Ndef(\sin).play;
Ndef(\sin).fadeTime = 2;
Ndef(\sin).stop

data.j

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	(

		Ndef (\sine, {
			//var freqs = [115, 105];
			Mix.new(Pan2.ar(
				Resonz.ar(PinkNoise.ar([
				caotico.linlin(0.1**17, 1, 1, 2),
				fijo.linlin(0.1**17, 1, 1, 2),
			], 10),
				complejo.linlin(0.1**11, 0.1, 220, 15000).poll,
				periodico.linlin(0.1**17, 0.7, 0.001, 0.015),
				),
				complejo.linlin(0.1**17, 0.9, -0.5, 0.5), 15));
		});
	)

},'/clase',recvPort: 5006);
)

		Ndef(\sine).play;
		Ndef(\sine).fadeTime = 5;
Ndef(\sine).free

///////////////////////////////////////////////////////////////////////////////////////
(
Ndef(\x,{
        arg freq1, freq2, amp=0.4;
        var sig;
        sig = SinOsc.aSr([freq1, freq1 - SinOsc.kr(1).range(1,7)],amp) + SinOsc.ar([freq2, freq2 + SinOsc.kr(10).range(1,7)], amp);
	sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.01), WhiteNoise.kr(200).range(10,10));
})
);

Ndef(\x).play
Ndef(\x).fadeTime = 2
Ndef(\x).set(\amp,0.1)
Ndef(\x).free

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	if(caotico > 0.7){Ndef(\x).set(\freq1,fijo.linlin(0.1**14, 1, 100, 200),\freq2,caotico.linlin(0.1**14, 1, 200, 300))};
	if(complejo > 0.7){Ndef(\x).set(\freq1, caotico.linlin(0.1**14, 1, 200, 300),\freq2,fijo.linlin(0.1**14, 1, 100, 200))};
    if(fijo > 0.7){Ndef(\x).set(\freq1,periodico.linlin(0.1**14, 1, 300, 400),\freq2,complejo.linlin(0.1**14, 1, 400, 500))};
	if(periodico > 0.7){Ndef(\x).set(\freq1,complejo.linlin(0.1**14, 1, 400, 500),\freq2,periodico.linlin(0.1**14, 1, 300, 400))};

},'/clase',recvPort: 5006);
)

///////////////////////////////////////////////////////////////////////////////////////

Ndef(\x).play
Ndef(\x).fadeTime = 8
Ndef(\x).set(\amp,0.9)
Ndef(\x).stop

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	if(caotico > 0.7){

		Ndef(\x,{|amp|
			var sig;
			sig = SinOsc.ar([complejo*600, caotico*600 - SinOsc.kr(1).range(1,7)],amp) + SinOsc.ar([caotico*600, periodico*600 + SinOsc.kr(10).range(1,7)], amp);
			sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.01), WhiteNoise.kr(20).range(10,10));
		})
	};

	if(complejo > 0.7){

		Ndef(\x,{|amp|
			var sig;
			sig = SinOsc.ar([complejo*300, caotico*300 - SinOsc.kr(1).range(1,7)],amp) + SinOsc.ar([fijo*300, complejo*800 + SinOsc.kr(10).range(1,7)], amp);
			sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.01), WhiteNoise.kr(20).range(10,10));
		})
	};

    if(fijo > 0.7){

		Ndef(\x,{|amp|
			var sig;
			sig = SinOsc.ar([complejo*200, periodico*200 - SinOsc.kr(1).range(1,7)],amp) + SinOsc.ar([caotico*200, periodico*200 + SinOsc.kr(100).range(1,7)], amp);
			sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.01), WhiteNoise.kr(20).range(10,10));
		})
	};

	if(periodico > 0.7){

		Ndef(\x,{|amp|
			var sig;
			sig = SinOsc.ar([fijo*500, periodico*500 - SinOsc.kr(1).range(1,7)],amp) + SinOsc.ar([periodico*500, fijo*500 + SinOsc.kr(10).range(1,7)], amp);
			sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.01), WhiteNoise.kr(20).range(10,10));
		})
	};

},'/clase',recvPort: 5006);
)

///////////////////////////////////////////////////////////////////////////////////////////
Ndef(\x).play
Ndef(\x).fadeTime = 5
Ndef(\x).set(\amp,0.5)
Ndef(\x).stop
h = Buffer.alloc(s, 8192, 1);

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	if(caotico > 0.7){

		Ndef(\x,{|amp|
			var sig;
			sig = SinOsc.ar([complejo*600, caotico*600 - SinOsc.kr(1).range(1,7)],amp,0.1) + SinOsc.ar([caotico*600, periodico*600 + SinOsc.kr(10).range(1,7)], amp);
			sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.001), WhiteNoise.kr(20).range(10,10), 0.5,0.5);
		})
	};

	if(complejo > 0.7){

		Ndef(\x,{|amp|
			var sig;
			sig = SinOsc.ar([complejo*300, caotico*300 - SinOsc.kr(1).range(1,7)],amp, 0.1) + SinOsc.ar([fijo*300, complejo*200 + SinOsc.kr(10).range(1,7)], amp);
			sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.01), WhiteNoise.kr(20).range(10,10), 0.5, 0.5);
		})
	};

    if(fijo > 0.7){

		Ndef(\x,
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
				newfreq = freq.lag(0.2);

				brown_noise = WhiteNoise.ar([0.5,0.5], amp*1.2);
				env = EnvGen.kr(Env.new([0, 1, 0.7, 0 ], [0.1, 0.5, 1.1], 1, 2, nil), onset);
				noise = brown_noise * env;
				sig = Resonz.ar(noise, newfreq*0.99, 0.0001);
				//noise = HPF.ar(noise, 440);
				//Pan2.ar(in*noise, LFSaw.kr(loudness.linlin(20, 45, 2, 0.01)), 1);
				//rev = Mix.ar(GVerb.ar(sig, room, 5, 0.8, 0.8, 10, 1));
				Normalizer.ar(DelayN.ar(sig, 10, 2.5), norm, 0.01);
	})};

	if(periodico > 0.7){

		Ndef(\x,
			{
				var in, freq, hasfreq, freq2, fft, amp, loudness, noise, norm;

				in = SoundIn.ar(0, MouseX.kr(0.001, 2));
				#freq, hasfreq = Pitch.kr(in, 60, 50, 10000, 1, 36).lag(0.5);
				#freq2, hasfreq = Pitch.kr(in, 60, 30, 8000, 100, 16).lag(0.5);
				fft = FFT(h, in);
				loudness = Loudness.kr(fft);
				amp = Amplitude.kr(in, 0.01, 2);
				noise = Resonz.ar(WhiteNoise.ar(freq.linexp(114, 1600, 1, 0.2), amp*0.1), freq2*1.34, 0.01, 1);
				noise = Pan2.ar(noise, LFSaw.kr(loudness.linexp(10, 45, 2, 0.01)), 1, 2,);
				Normalizer.ar(DelayN.ar(noise, 10, 2.5), norm, 0.01);
				//PanAz.ar(8, noise, 0, 1, 1);
	})};

},'/clase',recvPort: 5006);
)

///////////////////////////////////////////////////////////////////////////////////////////
Ndef(\x).play
Ndef(\x).fadeTime = 5
Ndef(\x).set(\amp,0.5)
Ndef(\x).stop
h = Buffer.alloc(s, 8192, 1);

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	if(caotico > 0.7){

		Ndef(\x,{|amp|
			var sig;
			sig = SinOsc.ar([complejo*600, caotico*600 - SinOsc.kr(1).range(1,7)],amp,0.1) + SinOsc.ar([caotico*600, periodico*600 + SinOsc.kr(10).range(1,7)], amp);
			sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.001), WhiteNoise.kr(20).range(10,10), 0.5,0.5);
		})
	};

	if(complejo > 0.7){

		Ndef(\x,{|amp|
			var sig;
			sig = SinOsc.ar([complejo*300, caotico*300 - SinOsc.kr(1).range(1,7)],amp, 0.1) + SinOsc.ar([fijo*300, complejo*200 + SinOsc.kr(10).range(1,7)], amp);
			sig = FreeVerb.ar(sig * amp, SinOsc.kr(0.01), WhiteNoise.kr(20).range(10,10), 0.5, 0.5);
		})
	};

    if(fijo > 0.7){

		Ndef(\x,
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
				newfreq = freq.lag(0.2);

				brown_noise = WhiteNoise.ar([0.5,0.5], amp*1.2);
				env = EnvGen.kr(Env.new([0, 1, 0.7, 0 ], [0.1, 0.5, 1.1], 1, 2, nil), onset);
				noise = brown_noise * env;
				sig = Resonz.ar(noise, newfreq*0.99, 0.0001);
				//noise = HPF.ar(noise, 440);
				//Pan2.ar(in*noise, LFSaw.kr(loudness.linlin(20, 45, 2, 0.01)), 1);
				//rev = Mix.ar(GVerb.ar(sig, room, 5, 0.8, 0.8, 10, 1));
				Normalizer.ar(DelayN.ar(sig, 10, 2.5), norm, 0.1);
	})};

	if(periodico > 0.99999999){

		Synth(\play, [\buf, ~samples[1..90].choose, \rate, (-5..5).choose.midiratio])

		};

},'/clase',recvPort: 5006);
)

///////////////////////////////////////////////////////////////////////////////////////////
Ndef(\x).play
Ndef(\x).fadeTime = 5
Ndef(\x).set(\amp,0.5)
Ndef(\x).stop
h = Buffer.alloc(s, 8192, 1);

(
OSCdef (\osc2, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	if(caotico > 0.99){
		Synth(\play, [\buf, ~fijo[0..~fijo.size].choose, \rate, (-5..5).choose.midiratio])
		};

	if((caotico > 0.8) && (caotico < 0.90)){
		Silent.ar(2)
		};

	if((caotico > 0.20) && (caotico < 0.80)){
		Synth(\play, [\buf, ~fijo[0..~fijo.size].choose, \rate, (-8..0).choose.midiratio]);
		("me lanzaron ca!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1".postln)
		};

	if(complejo > 0.99999999){
		Synth(\play, [\buf, ~periodico[0..~periodico.size].choose, \rate, (-5..5).choose.midiratio])

		};

    if(fijo > 0.9999999){

		Synth(\play2, [\buf, ~caotico[0..~caotico.size].choose, \rate, (-5..5).choose.midiratio])

		};

	if(periodico > 0.9){

		Synth(\play3, [\buf, ~fijo[1..~fijo.size].choose, \rate, (-5..5).choose.midiratio])

		};

},'/clase',recvPort: 5006);
)

Synth(\play1).stop