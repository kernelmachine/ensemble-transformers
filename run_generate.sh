#!/bin/bash

# important! make sure to set num_tasks_per_node to 1!

NUM_GPUS_PER_NODE=$1
SLURM=$2

if [ $SLURM == "True" ]; then
    GPUS_PER_NODE=$NUM_GPUS_PER_NODE
    NNODES=$SLURM_JOB_NUM_NODES
    WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

    if [ $NNODES == "1" ]; then
        RANDOM_INT=$(shuf -i 1024-65000 -n 1)
        PORT=$(($RANDOM))
        C10_PORT_PHRASE="--master_port $PORT"
    else
        C10_PORT_PHRASE="--master_port 29500"
    fi;
    echo $SLURM_PROCID

    export LAUNCHER=" \
        torchrun \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --master_addr $MASTER_ADDR \
        --node_rank ${SLURM_PROCID} \
        $C10_PORT_PHRASE \
        "
else 
    NNODES=1
    GPUS_PER_NODE=1
    WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
    export LAUNCHER="python -u"
fi;

export CMD="generate.py"



export EXEC="${LAUNCHER} $CMD"

echo $EXEC
bash -c "$EXEC"
