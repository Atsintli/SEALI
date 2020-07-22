(
//n = NetAddr("127.0.0.1"); // local machine
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.8],[SpecFlatness],
	[OnsetStatistics]];
var ventaneo = 21;
t = ~startRec.("/Users/Atsintli/Desktop/recs", 1.5, {|fileName|
    Task({
       // 0.05.wait;
        ~res = ~getAudioFeatures.([[fileName]], nil, features, ~ventanizar, ventaneo);
        ~data = ~res[\unknown].flatten.flatten;
    }).play
});

~client = NetAddr("127.0.0.1", 5005); // loopback ----

Task({
	inf.do({
		~client.sendMsg("/features", *~data);
		0.2.wait;
	});
}).play;
)
~data.size
/////////////////////////////////////////////////

(
//n = NetAddr("127.0.0.1"); // local machine
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.8],[SpecFlatness],
	[BeatStatistics]];
var ventaneo = 41;
t = ~startRec.("/Users/Atsintli/Desktop/recs", 1.3, {|fileName|
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
		0.3.wait;
	});
}).play;
)

////////////////////////////////////////////////////////////////////////////////////////////////

(
//n = NetAddr("127.0.0.1"); // local machine
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.8]];
var ventaneo = 82;
t = ~startRec.("/Users/Atsintli/Desktop/recs", 3, {|fileName|
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
//////////////////////////////////versión de 4 segundos/////////////////////////////////////////////////////

(
//n = NetAddr("127.0.0.1"); // local machine
var features = [[Chromagram],[SpecPcile, 0.95],[SpecPcile, 0.8],[SpecFlatness],
	[OnsetStatistics]];
var ventaneo = 328;
t = ~startRec.("/Users/Atsintli/Desktop/recs", 8.1, {|fileName|
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
		0.1.wait;
	});
}).play;
)
