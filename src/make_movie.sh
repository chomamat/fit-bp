#!/bin/bash
if [ "$#" -ne 3 ]; then
	echo "Illegal number of arguments! [frame_rate][folder][output]"
    exit 1
fi

ffmpeg -r $1 -pattern_type glob -i "$2/*.png" -c:v libx264 -tune stillimage -y $3