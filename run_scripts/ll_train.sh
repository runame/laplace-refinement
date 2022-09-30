#!/bin/bash
#SBATCH -p gpu-2080ti                     # partition
#SBATCH --gres=gpu:rtx2080ti:1            # type and number of gpus
#SBATCH --time=72:00:00                   # job will be cancelled after max. 72h
#SBATCH --output=ll_train_%A_%a.out
#SBATCH --array=1-5

# print info about current job
scontrol show job $SLURM_JOB_ID

declare -a datasets=("fmnist" "cifar10" "cifar100")
declare -a chain_ids=(1 2 3)

for dataset in "${datasets[@]}";
do
    # MAP
    python ll_train.py --method map --dataset $dataset --randseed $SLURM_ARRAY_TASK_ID

    # HMC
    for chain_id in "${chain_ids[@]}";
    do
        python ll_train.py --method hmc --n_burnins 100 --n_samples 200 --dataset $dataset --randseed $SLURM_ARRAY_TASK_ID --chain_id $chain_id
    done

    # NF+N(0,I)
    python ll_train.py --method nf_naive --flow_type radial --n_flows 1 --dataset $dataset --randseed $SLURM_ARRAY_TASK_ID

    for n_flows in {5..30..5};
    do
        echo $n_flows
        python ll_train.py --method nf_naive --flow_type radial --n_flows $n_flows --dataset $dataset --randseed $SLURM_ARRAY_TASK_ID
    done


    # Refine
    python ll_train.py --method refine --flow_type radial --n_flows 1 --dataset $dataset --randseed $SLURM_ARRAY_TASK_ID

    for n_flows in {5..30..5};
    do
        echo $n_flows
        python ll_train.py --method refine --flow_type radial --n_flows $n_flows --dataset $dataset --randseed $SLURM_ARRAY_TASK_ID
    done
done
