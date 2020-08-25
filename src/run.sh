#!/bin/bash

# python script, model, total_edge, local_edge,core per local edge, log dir

pyscript=${1}
_model=${2}
_total_edge=${3}
edge=${4}
core=${5}
_log=${6}

_lr=0.01
_host="localhost"
_port=6379
_ratio=0.5

# flush redis
redis-cli -h ${_host} -p ${_port} FLUSHDB
#redis-cli -h ${_host} -p ${_port} PING


total_cores=$(cat /proc/cpuinfo | grep "processor" | wc -l)
if ((total_cores <= edge*core)) ; then
    echo "Required cores exceed total logical cores!"
    # exit 1
fi

# generate cpu cores sets
GPUNUM=1
stidx=0
coreset=()
for ((i=1;i <= edge; i++));
do
    edx=$((stidx + core - 1))
    if ((core > 1)) ; then
        coreset[i]="${stidx}-${edx}"
    else
        coreset[i]="${stidx}"
    fi
    stidx=$((edx + 1))
done

#for cs in ${coreset[@]}
#do
#    echo ${cs}
#done


for idx in $(seq 1 ${#coreset[@]}) ; do

    #taskset --cpu-list ${coreset[idx]} python ${pyscript} --dataset "/workspace" --model ${_model} --lr ${_lr} --lrscheduler --host ${_host} --port ${_port} --edgenum ${_total_edge} --gpu --gpuindex $((idx % GPUNUM)) --fixedratio ${_ratio} &
    taskset --cpu-list ${coreset[idx]} python ${pyscript} --dataset "/workspace" --model ${_model} --lr ${_lr} --lrscheduler --host ${_host} --port ${_port} --edgenum ${_total_edge} --gpu --gpuindex $((idx % GPUNUM)) --fixedratio ${_ratio} --noncriticalremove &
 
done
wait

if [ -d ${_log} ]; then
    mv ./2020*.log ${_log}
else
    mkdir -p ${_log}
    mv ./2020*.log ${_log}
fi

echo "Finished"
