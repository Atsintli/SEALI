(
~currentPath = thisProcess.nowExecutingPath.dirname;
(~currentPath++"/ventanizar.scd").load;
(~currentPath++"/rec-loop.scd").load;
(~currentPath++"/stringify.scd").load;
(~currentPath++"/get-audio-features.scd").load;
s.boot;
)


