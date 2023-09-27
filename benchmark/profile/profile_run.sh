#!/bin/bash

export MASTER_ADDR=`scontrol show hostname $SLURM_NODELIST| head -n 1`
export MASTER_PORT=5566


if [ $2 -eq 6144 ]; then
  num_layer=2
else
  num_layer=4
fi

export TP=$3
export DP=$4

python -um oases.profile.profiler \
                      --batch_size $1 \
                      --hidden_size $2 \
                      --num_layer $num_layer




