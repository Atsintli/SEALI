//n = NetAddr("127.0.0.1"); // local machine
(
OSCdef (\osc, {|msg, time, addr, recvPort|
	var clase, caotico, complejo, fijo, periodico;
	msg.debug("=====================");
	# clase, caotico, complejo, fijo, periodico = msg;

	SynthDef (\prueba, {|freq|
		var sig, out;
		sig = SinOsc.ar(freq, 0, 0.5);
		out = Out.ar(0, sig);
	}).add;

Synth (\prueba, [\freq, caotico]);

},'/clase',recvPort: 5006); // once only
)

~client = NetAddr("127.0.0.1", 5005); // loopback ----

//cada que hagas un analisis
~client.sendMsg("/features", *~data);

data.j

(
Task({
	inf.do({
		~client.sendMsg("/features", *~data);
		0.1.wait;
	});
}).play;
)