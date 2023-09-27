#!/bin/bash
#SBATCH -o sbatch.out
#SBATCH -N 8
#SBATCH --ntasks-per-node 4

# Please modify the sbatch settings above
################################
# The output filename:
SAVE_NAME=profile_res.pt
################################

OUT_NAME=profile.out
NNODES=$SLURM_NNODES
mkdir -p profile_log/outputs/
cat /dev/null > profile_log/${OUT_NAME}
echo `scontrol show hostname $SLURM_NODELIST` >> profile_log/${OUT_NAME}

#################
# The profile settings
MAX_HIDDEN=12288
MAX_BS=128
MAX_TP=$SLURM_NTASKS
##################

for ((tp=1; tp<=MAX_TP; tp*=2))
do  
    for ((bs=1; bs<=MAX_BS; bs*=2))
    do
        for ((hidden=1024; hidden<=MAX_HIDDEN; hidden+=1024))
        do  
            dp=`expr ${NNODES} \* 4 / ${tp}`
            echo "using tp_degree ${tp} dp_degree ${dp} batch_size ${bs} hidden_size ${hidden}" >> profile_log/${OUT_NAME}
            srun bash profile_run.sh $bs $hidden $tp $dp 2>&1 | tee profile_log/outputs/batch_${bs}_hidden_${hidden}_tp_${tp}.out
            sleep 5
            srun killall python
            sleep 5
            grep  "Rank 0 " profile_log/outputs/batch_${bs}_hidden_${hidden}_tp_${tp}.out >> profile_log/${OUT_NAME}
            grep  "CUDA out of memory." profile_log/outputs/batch_${bs}_hidden_${hidden}_tp_${tp}.out >> profile_log/${OUT_NAME}
            echo "==================" >> profile_log/${OUT_NAME}
        done
    done
done

python -m oases.profile.result_parser --log_path=profile_log/${OUT_NAME} --save_name=./${SAVE_NAME} --max_bs $MAX_BS --max_hidden $MAX_HIDDEN

