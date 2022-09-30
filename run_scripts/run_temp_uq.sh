#!/bin/bash
#SBATCH -p gpu-2080ti                     # partition
#SBATCH --gres=gpu:rtx2080ti:1            # type and number of gpus
#SBATCH --time=72:00:00                   # job will be cancelled after max. 72h
#SBATCH --output=temp_uq_%A_%a.out
#SBATCH --array=1-5

# print info about current job
scontrol show job $SLURM_JOB_ID

declare -a fmnist_datasets=("FMNIST-OOD" "R-FMNIST")
declare -a cifar_datasets=("CIFAR-10-OOD" "CIFAR-10-C" "CIFAR-100-OOD")
# Set the path to your data and models directories here.
data_root=/mnt/qb/hennig/data
models_root=/mnt/qb/hennig/pretrained_models

for dataset in "${fmnist_datasets[@]}";
do
    # Assuming you have activated your conda environment
    python uq.py --benchmark $dataset --method map --use_temperature_scaling True --model LeNet --data_root ${data_root} --models_root ${models_root} --model_seed $SLURM_ARRAY_TASK_ID
done

for dataset in "${cifar_datasets[@]}";
do
    # Assuming you have activated your conda environment
    python uq.py --benchmark $dataset --method map --use_temperature_scaling True --model WRN16-4 --data_root ${data_root} --models_root ${models_root} --model_seed $SLURM_ARRAY_TASK_ID
done
