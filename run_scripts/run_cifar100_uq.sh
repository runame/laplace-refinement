#!/bin/bash
#SBATCH -p gpu-v100                       # partition
#SBATCH --gres=gpu:v100:1                 # type and number of gpus
#SBATCH --time=72:00:00                   # job will be cancelled after max. 72h
#SBATCH --output=cifar100_uq_%A_%a.out
#SBATCH --array=1-5

# print info about current job
scontrol show job $SLURM_JOB_ID

# Set the path to your data and models directories here.
data_root=/mnt/qb/hennig/data/
models_root=./pretrained_models

declare -a methods=("map" "hmc")
for method in "${methods[@]}";
do
    echo ${method}
    python uq.py --benchmark CIFAR-100-OOD --method ${method} --prior_precision 40 --model WRN16-4 --data_root ${data_root} --models_root ${models_root} --model_seed $SLURM_ARRAY_TASK_ID --compute_mmd
done

declare -a prior_optim=("marglik" "CV")
for prior in "${prior_optim[@]}";
do
    # LA-NN-MC
    python uq.py --benchmark CIFAR-100-OOD --method laplace --subset_of_weights last_layer --hessian_structure diag --optimize_prior_precision $prior --pred_type nn --link_approx mc --n_samples 20 --model WRN16-4 --data_root ${data_root} --models_root ${models_root} --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --run_name CIFAR-100-OOD/la_nn_mc_${prior}_$SLURM_ARRAY_TASK_ID

    # LA-MC
    python uq.py --benchmark CIFAR-100-OOD --method laplace --subset_of_weights last_layer --hessian_structure diag --optimize_prior_precision $prior --pred_type glm --link_approx mc --n_samples 20 --model WRN16-4 --data_root ${data_root} --models_root ${models_root} --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --run_name CIFAR-100-OOD/la_glm_mc_${prior}_$SLURM_ARRAY_TASK_ID

    # LA-Probit
    python uq.py --benchmark CIFAR-100-OOD --method laplace --subset_of_weights last_layer --hessian_structure diag --optimize_prior_precision $prior --pred_type glm --link_approx probit --model WRN16-4 --data_root ${data_root} --models_root ${models_root} --compute_mmd --model_seed $SLURM_ARRAY_TASK_ID --run_name CIFAR-100-OOD/la_glm_probit_${prior}_$SLURM_ARRAY_TASK_ID
done

declare -a nfmethods=("refine" "nf_naive")
for nfmethod in "${nfmethods[@]}";
do
    echo 1
    python uq.py --benchmark CIFAR-100-OOD --method ${nfmethod}_radial_1 --prior_precision 40 --model WRN16-4 --data_root ${data_root} --models_root ${models_root} --model_seed $SLURM_ARRAY_TASK_ID --compute_mmd
    for n_flows in {5..30..5};
    do
        echo $n_flows
        python uq.py --benchmark CIFAR-100-OOD --method ${nfmethod}_radial_${n_flows} --prior_precision 40 --model WRN16-4 --data_root ${data_root} --models_root ${models_root} --model_seed $SLURM_ARRAY_TASK_ID --compute_mmd
    done
done

# Baselines

python uq.py --benchmark CIFAR-100-OOD --method ensemble --nr_components 5 --model WRN16-4 --data_root ${data_root} --models_root ${models_root} --model_seed $SLURM_ARRAY_TASK_ID

python uq.py --benchmark CIFAR-100-OOD --method bbb --model WRN16-4-BBB-flipout --normalize --data_root ${data_root} --models_root ${models_root} --model_seed $SLURM_ARRAY_TASK_ID

python uq.py --benchmark CIFAR-100-OOD --method csghmc --model WRN16-4-CSGHMC --normalize --data_root ${data_root} --models_root ${models_root} --model_seed $SLURM_ARRAY_TASK_ID
