(
o = Server.local.options;
//Server.local.options.device = "Jack";
Server.local.options.memSize = 2.pow(18);
Server.internal.options.memSize = 2.pow(18);
s.reboot;
)

(
~currentPath = thisProcess.nowExecutingPath.dirname;
(~currentPath++"/ventanizar.scd").load;
(~currentPath++"/rec-loop.scd").load;
(~currentPath++"/stringify.scd").load;
(~currentPath++"/get-audio-features.scd").load;
//(~currentPath++"/get_audios_2.scd").load;
//(~currentPath++"/OSC_Client.scd").load;
)
