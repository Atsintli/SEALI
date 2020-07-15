(
//n = NetAddr("127.0.0.1"); // local machine
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.80],[SpecFlatness],[BeatStatistics]];
var ventaneo = 82;
t = ~startRec.("~/Desktop/recs", 2.4, {|fileName|
    Task({
		0.1.wait;
        ~res = ~getAudioFeatures.([[fileName]], nil, features, ~ventanizar, ventaneo);
        ~data = ~res[\unknown].flatten.flatten;
		~client.sendMsg("/features", *~data);
    }).play
});

~client = NetAddr("127.0.0.1", 5005); // loopback ----
)

~data.size

t.stop