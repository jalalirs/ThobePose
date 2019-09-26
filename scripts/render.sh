#!/bin/bash

### Run rendering in batches.

source ~/.bashrc
framesperrun=50
from=1
to=1193

for i in $(seq $from $framesperrun $to)
  do
    start=$i
    end=$(($i + $framesperrun))
    echo "$start $end"
    if [ "$end" -gt "1193" ]
    then
    	end=1193
    fi
    #mayapy ./render_thobe.py $start $end    # for thobe character 
    mayapya ./render_plain.py $start $end  # for plain character 
 done
