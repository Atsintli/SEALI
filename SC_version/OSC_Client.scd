(
//n = NetAddr("127.0.0.1"); // local machine
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],
	[OnsetStatistics]];
var ventaneo = 82;
t = ~startRec.("~/Desktop/recs", 2, {|fileName|
    Task({
        0.05.wait;
        ~res = ~getAudioFeatures.([[fileName]], nil, features, ~ventanizar, ventaneo);
        ~data = ~res[\unknown].flatten.flatten;
    }).play
});

~client = NetAddr("127.0.0.1", 5005); // loopback ----

Task({
	inf.do({
		~client.sendMsg("/features", *~data);
		0.7.wait;
	});
}).play;
)

~data.size
