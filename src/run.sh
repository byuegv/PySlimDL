#!/bin/bash
# flush redis
redis-cli flushall

_edge=${2}
_log=${3}

#for idx in $(seq 1 ${_edge}) ; do

  sleep 10s && python ${1} --dataset "/workspace" --edgenum ${_edge} --gpu --fixedratio "0.7" &

  sleep 5s && python ${1} --dataset "/workspace" --edgenum ${_edge} --gpu  --fixedratio "0.7" &

  sleep 7s && python ${1} --dataset "/workspace" --edgenum ${_edge} --noncriticalremove --fixedratio "0.3" &

  sleep 13s && python ${1} --dataset "/workspace" --edgenum ${_edge}  --noncriticalremove --fixedratio "0.3" &

#done
wait

if [ -d ${_log} ]; then
    mv ./2020*.log ${_log}
else
    mkdir -p ${_log}
    mv ./2020*.log ${_log}
fi

echo "Finished"
