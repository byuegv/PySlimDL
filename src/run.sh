#!/bin/bash

_edge=${2}
_model=${3}
_loop=${4}

_log=${5}

_lr=0.01
_host="localhost"
_port=6379
_ratio=0.5

# flush redis
redis-cli -h ${_host} -p ${_port} FLUSHDB
#redis-cli -h ${_host} -p ${_port} PING


for idx in $(seq 1 ${_loop}) ; do

  python ${1} --dataset "/workspace" --model ${_model} --lr ${_lr} --lrscheduler --host ${_host} --port ${_port} --edgenum ${_edge} --gpu --gpuindex 0 --fixedratio ${_ratio} &
  python ${1} --dataset "/workspace" --model ${_model} --lr ${_lr} --lrscheduler --host ${_host} --port ${_port} --edgenum ${_edge} --gpu --gpuindex 0 --fixedratio ${_ratio} --noncriticalremove &
 
done
wait

if [ -d ${_log} ]; then
    mv ./2020*.log ${_log}
else
    mkdir -p ${_log}
    mv ./2020*.log ${_log}
fi

echo "Finished"
