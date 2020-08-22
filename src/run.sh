#!/bin/bash
redis-cli flushall
_edge=${2}
#for idx in $(seq 1 ${_edge}) ; do

  python ${1} --dataset "/workspace" --edgenum ${_edge} --gpu --noncriticalremove --fixedratio "0.7" &
  sleep 10s

  python ${1} --dataset "/workspace" --edgenum ${_edge}  --noncriticalremove --fixedratio "0.7" &
  #sleep ${idx}s
  sleep 5s

  python ${1} --dataset "/workspace" --edgenum ${_edge} --gpu --noncriticalremove --fixedratio "0.7" &
  sleep 10s

  python ${1} --dataset "/workspace" --edgenum ${_edge}  --noncriticalremove --fixedratio "0.7" &
  #sleep ${idx}s

 # python ${1} --dataset "/workspace" --edgenum ${_edge} --gpu --noncriticalremove --fixedratio "0.7" &
  
#done
wait

echo "Finished"
