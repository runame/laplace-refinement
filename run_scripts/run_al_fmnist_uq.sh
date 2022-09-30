#!/bin/bash
#SBATCH -p gpu-v100                       # partition
#SBATCH --gres=gpu:v100:1                 # type and number of gpus
#SBATCH --time=72:00:00                   # job will be cancelled after max. 72h
#SBATCH --output=fmnist_al_uq_%A_%a.out
#SBATCH --array=1-5

# print info about current job
scontrol show job $SLURM_JOB_ID

declare -a datasets=("FMNIST-OOD" "R-FMNIST")
declare -a prior_optim=("marglik" "CV")
# Set the path to your data and models directories here.
data_root=/mnt/qb/hennig/data/
models_root=/mnt/qb/hennig/pretrained_models

for dataset in "${datasets[@]}";
do
    # Assuming you have activated your conda environment
     # MAP
    python uq.py --benchmark $dataset --method map --model FMNIST-MLP --data_root ${data_root} --models_root ${models_root} --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --run_name $dataset/al_map_$SLURM_ARRAY_TASK_ID

    # HMC
    python uq.py --benchmark $dataset --method hmc --subset_of_weights all --prior_precision 510 --model FMNIST-MLP --data_root ${data_root} --models_root ${models_root} --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --run_name $dataset/al_hmc_$SLURM_ARRAY_TASK_ID

    for prior in "${prior_optim[@]}";
    do
        # LA-NN-MC
        python uq.py --benchmark $dataset --method laplace --subset_of_weights all --hessian_structure diag --optimize_prior_precision $prior --pred_type nn --link_approx mc --n_samples 20 --model FMNIST-MLP --data_root ${data_root} --models_root ${models_root} --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --run_name $dataset/al_la_nn_mc_${prior}_$SLURM_ARRAY_TASK_ID

        # LA-MC
        python uq.py --benchmark $dataset --method laplace --subset_of_weights all --hessian_structure diag --optimize_prior_precision $prior --pred_type glm --link_approx mc --n_samples 20 --model FMNIST-MLP --data_root ${data_root} --models_root ${models_root} --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --run_name $dataset/al_la_glm_mc_${prior}_$SLURM_ARRAY_TASK_ID

        # LA-Probit
        python uq.py --benchmark $dataset --method laplace --subset_of_weights all --hessian_structure diag --optimize_prior_precision $prior --pred_type glm --link_approx probit --model FMNIST-MLP --data_root ${data_root} --models_root ${models_root} --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --run_name $dataset/al_la_glm_probit_${prior}_$SLURM_ARRAY_TASK_ID
    done
    
    echo 1
    python uq.py --benchmark $dataset --method refine_radial_1 --model LeNet --data_root /mnt/qb/hennig/data/ --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --subset_of_weights all

    for n_flows in {5..100..5};
    do
        echo $n_flows
        python uq.py --benchmark $dataset --method refine_radial_$n_flows --model LeNet --data_root /mnt/qb/hennig/data/ --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --subset_of_weights all
    done
done
