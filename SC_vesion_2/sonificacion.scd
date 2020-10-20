(
o = Server.local.options;
//Server.local.options.device = "Jack";
Server.local.options.memSize = 2.pow(21);
Server.internal.options.memSize = 2.pow(21);
s.reboot;
)

(
~currentPath = thisProcess.nowExecutingPath.dirname;
(~currentPath++"/ventanizar.scd").load;
(~currentPath++"/rec-loop.scd").load;
(~currentPath++"/stringify.scd").load;
(~currentPath++"/get-audio-features.scd").load;
//(~currentPath++"/get_audios_syths_movil_ii.scd").load;
//(~currentPath++"/OSC_Client.scd").load;
)


(
h = Buffer.alloc(s, 8192, 1);
//Ndef(\sin).play;
//Ndef(\sin).fadeTime=2;
Ndef(\buf).fadeTime=5;
Ndef(\buf).play
)

(
//n = NetAddr("127.0.0.1"); // local machine
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness]];
var ventaneo = 1;
t = ~startRec.("~/Desktop/recs", 2, {|fileName|
	Task({
		0.1.wait;
		~res = ~getAudioFeatures.([[fileName]], nil, features, ~ventanizar, ventaneo);
		~data = ~res[\unknown].flatten.flatten;
		~client.sendMsg("/features", *~data);
	}).play
});

~client = NetAddr("127.0.0.1", 5005); // loopback ----
)

(
OSCdef (\osc2, {|msg, time, addr, recvPort|
	var clase, ataques, cress, largos, tremolando;
	# clase, ataques, cress, largos, tremolando = msg;


	if(ataques > 0.99){
		(
			{

				~buffselect = ~ataques[0..~ataques.size].choose;

				50.do{arg i;
					var timeprop = (i/199.0)**3;

					Synth(\sfgrain2,[
						\bufnum, ~buffselect,
						\startPos,rrand(0.0,timeprop),
						\dur, exprand(0.004, 0.1),
						\atack, exprand(0.5, 1),
						\amp, exprand(0.005,0.07),
						\pan, 1.0.rand2]);
					rrand((timeprop*0.1).max(0.1),timeprop*0.4).wait
				};
			}.fork
		)
	};

	if ((ataques > 0.5) && (ataques < 0.9),{
		"Ataques con soundfreqgrain".postln;
		if (cress > 0.01)
		{~lastval= exprand(20,6)}{~lastval = exprand(5,0.06)};

		{
			~buffselect = ~largos[0..~largos.size].choose;

			50.do{arg i;
				var prop, timestart, timeend;
				prop= (i%200)/200;
				timestart= prop*0.8;
				timeend= prop*(0.8+(0.1*~lastval));
				Synth(\sfgrain,[
					\bufnum, ~buffselect,
					\attack, exprand(0.05,1),
					\startPos,rrand(timestart,timeend),
					\amp, exprand(0.005,0.01),
					\pan, ~lastval.rand2,
					\dur, 1+(~lastval*0.5)]);
				//max in this to avoid ever going near 0.0 wait time, which would crash the computer!
				exprand((~lastval*0.05)+(0.01).max(0.1), ~lastval*(1)).wait
			};
		}.fork;
	};
	);

	if((ataques > 0.1) && (ataques < 0.4),{
		(
			(
				Synth(\sines,
					[\vel, largos.linexp(370, 780, 0.09, 0.01), \vel2, largos.linexp(0, 780, 0.09, 0.01)],
				);
			)
		)
	}
	);

	if((cress > 0.6) && (cress < 1),{
	"fui activadoooooo sinegrain2".postln
		(
			{
				30.do{|i|
					var timeprop = (i/100.0)**4;
					Synth(\sinegrain2,[
						\freq, exprand(largos*3000+(4*i-1),largos*5001+(8*i)),
						\amp, exprand(0.1,0.09),
						//\atack, exprand(0.1, 0.9),
						\pan, exprand(0.025, 0.001)]);
					rrand((timeprop*0.1).max(0.01),timeprop*0.01).wait;
					//rrand((timeprop*0.01).max(0.01),timeprop*0.01).wait
				};
			}.fork;
		)
	}
	);

	if((largos > 0.32) && (largos < 1),{
		"buffer de largos".postln;
		Synth(\largos, [
			\buf, ~largos[0..~largos.size].choose,
			\pan, ataques,
			\rate, (-24..12).choose.midiratio])
	}
	);

	if((tremolando > 0.32) && (tremolando < 1),{
		"buffer de largos".postln;
		Synth(\largos, [
			\buf, ~tremolando[0..~tremolando.size].choose,
			\pan, tremolando,
			\rate, (-24..-12).choose.midiratio])
	}
	);

},'/clase',recvPort: 5006);
)

