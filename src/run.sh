#!/bin/bash
redis-cli flushall
_edge=${1}
#for idx in $(seq 1 ${_edge}) ; do

  python main.py --dataset "/workspace" --edgenum ${_edge} --gpu --noncriticalremove --fixedratio "0.7" &
  sleep 10s

  python main.py --dataset "/workspace" --edgenum ${_edge}  --noncriticalremove --fixedratio "0.7" &
  #sleep ${idx}s
  sleep 5s

  python main.py --dataset "/workspace" --edgenum ${_edge} --gpu --noncriticalremove --fixedratio "0.7" &
  sleep 10s

  python main.py --dataset "/workspace" --edgenum ${_edge}  --noncriticalremove --fixedratio "0.7" &
  #sleep ${idx}s

 # python main.py --dataset "/workspace" --edgenum ${_edge} --gpu --noncriticalremove --fixedratio "0.7" &
  
#done
wait

echo "Finished"
