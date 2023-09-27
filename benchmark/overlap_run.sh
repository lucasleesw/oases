#!/bin/bash
###############################       
# Slurm related environments
MASTER_ADDR=`scontrol show hostname $SLURM_NODELIST| head -n 1`
MASTER_PORT=5566
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
###############################       
# GPU number per node
GPUS_PER_NODE=4
###############################       
# Model settings
bs=4
hidden=3072
layer=24
###############################       
# Init parallel degree settings, might be changed with Oases planner
export TP=4
export DP=2
###############################
# Set training schedule, available train schedules are:
# default_checkpoint
# optim_comm_checkpoint 
# overlap_checkpoint
# oases
train_schedule=oases
###############################       
# If using Oases planner, please finish profiling (./profile/sbatch_profile.sh) and add:
# --auto_plan --profile_path './profile/profile_res.pt'
###############################              
# torch launch
python -um torch.distributed.launch --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
                                   --node_rank $NODE_RANK --master_addr $MASTER_ADDR \
                                   --master_port $MASTER_PORT \
                                   overlap_tp.py \
                                   --batch_size $bs \
                                   --num_layer $layer \
                                   --schedule_option $train_schedule \
                                   --hidden_size $hidden \
                                   --gas 20

