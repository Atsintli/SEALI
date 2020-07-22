~path = thisProcess.nowExecutingPath.dirname +/+"datasegments_82.csv"

//x = CSVFileReader.read(~path, false).postcs;
x = CSVFileReader.read(~path, false);
~size = x.size

(
t = Task({
	~size.do{|i|
		~cont = i + 1;
		~client.sendMsg("/features", *x[i].asFloat); //max array size is 1634
		0.0125.wait;
		~cont.postln;
}}).play;
~client = NetAddr("127.0.0.1", 5005); // loopback ----
)

t.stop