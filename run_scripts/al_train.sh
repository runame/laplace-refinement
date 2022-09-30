#!/bin/bash
#SBATCH -p gpu-2080ti                     # partition
#SBATCH --gres=gpu:rtx2080ti:1            # type and number of gpus
#SBATCH --time=72:00:00                   # job will be cancelled after max. 72h
#SBATCH --output=al_train_%A_%a.out
#SBATCH --array=1-5

# print info about current job
scontrol show job $SLURM_JOB_ID

declare -a chain_ids=(2 3)

# MAP
python al_train.py --method map --dataset fmnist --randseed $SLURM_ARRAY_TASK_ID

# HMC
for chain_id in "${chain_ids[@]}";
do
python al_train.py --method hmc --n_burnins 100 --n_samples 200 --dataset fmnist --randseed $SLURM_ARRAY_TASK_ID --chain_id $chain_id
done

# Refine
python al_train.py --method refine --flow_type radial --n_flows 1 --dataset fmnist --randseed $SLURM_ARRAY_TASK_ID
#
for n_flows in {5..100..5};
do
    echo $n_flows
    python al_train.py --method refine --flow_type radial --n_flows $n_flows --dataset fmnist --randseed $SLURM_ARRAY_TASK_ID
done
