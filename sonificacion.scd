(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

			// segment the data: split into different parameters
			c = case
				{caotico > 0.7} {
					//"send to param 1: ".post; i.postln;
					Ndef(\sin).play;
					Ndef(\sin).fadeTime = 1;
					Ndef (\sin, {SinOsc.ar(MouseX.kr([120,121], [18000,18100], 2, 1,0,0))});
				}

				{complejo > 0.7} {
					//"send to param 2: ".post; i.postln;
					Ndef (\sin, {SinOsc.ar(XLine.kr([18000,18100], [120,121], 1, 1,0,0))});
				}

				{fijo > 0.7} {
					//"send to param 3: ".post; i.postln;
					Ndef (\sin, {

						SinOsc.ar(440)}

					);
			}
				{periodico > 0.7} {
					//"send to param 3: ".post; i.postln;
					Ndef (\sin, {SinOsc.ar(XLine.kr([200,2000], [120,121], 1, 1,0,0))}

					);
};

~dataMap.value;

},'/clase',recvPort: 5006); // once only
)

////////////////////////////////////////////////////////////////////////////////////////////////

(
Task({
	inf.do({
		~client.sendMsg("/features", *~data);
		0.5.wait;
	});
}).play;
)

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	//msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	(
		Ndef(\sin).play;
		Ndef(\sin).fadeTime = 1;
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

////////////////////////////////////////////////////////////////////////////////

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	//msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	(
		Ndef(\sin).play;
		Ndef(\sin).fadeTime = 2;
		Ndef (\sin, {
			//var freqs = [115, 105];
			Mix.new(Pan2.ar(SinOsc.ar([
				caotico.linexp(0.1**17, 1, 222, 6013),
				complejo.linexp(0.1**17, 1, 309, 1009),
				fijo.linexp(0.1**17, 1, 305, 380),
				periodico.linexp(0.1**17, 1, 513, 6000),
			], 0, 0.01), complejo.linexp(0.1**17, 0.9, -0.5, 0.5), 0.2));
		});
	)

},'/clase',recvPort: 5006); // once only
)

data.j

(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	(
		Ndef(\sin).play;
		Ndef(\sin).fadeTime = 5;
		Ndef (\sin, {
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


Ndef(\sin).free

///////////////////////////////////////////////////////////////////////////////////////
(
Ndef(\x,{
        arg freq1, freq2, amp=0.4;
        var sig;
        sig = SinOsc.ar([freq1, freq1 - SinOsc.kr(1).range(1,7)],amp) + SinOsc.ar([freq2, freq2 + SinOsc.kr(10).range(1,7)], amp);
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
Ndef(\x).fadeTime = 1
Ndef(\x).set(\amp,0.01)
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
