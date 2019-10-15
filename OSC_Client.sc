(
o = Server.local.options;
Server.local.options.memSize = 2.pow(18);
Server.internal.options.memSize = 2.pow(18);
s.boot;
)

//n = NetAddr("127.0.0.1"); // local machine
(
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],
	[BeatStatistics]];
var ventaneo = 20;
t = ~startRec.("~/Desktop/recs", 0.5, {|fileName|
    Task({
        0.5.wait;
        ~res = ~getAudioFeatures.([[fileName]], nil, features, ~ventanizar, ventaneo);
        ~data = ~res[\unknown].flatten.flatten;
    }).play
});

~client = NetAddr("127.0.0.1", 5005); // loopback ----

Task({
	inf.do({
		~client.sendMsg("/features", *~data);
		0.5.wait;
	});
}).play;
)