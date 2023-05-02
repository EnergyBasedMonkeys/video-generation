#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_2023sp_02

## change the partition number to use different number of GPUs

#SBATCH --partition=n1s16-v100-2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=40G

#SBATCH --time=15:00:00
#SBATCH --output=segnet-train.out
#SBATCH --error=segnet-train.err
#SBATCH --exclusive
#SBATCH --requeue


mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL23SP/dataset_v2.sqsh /tmp
echo "Dataset is copied to /tmp"

singularity exec --nv \
--bind /scratch \
--overlay /tmp/dataset_v2.sqsh:/input_data \
/share/apps/images/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
/bin/bash -c "
python3 -m pip install /scratch/<user>/video-generation
transofmer train --output-dir /scratch/<user>/transformer_out --dataset-dir ./dataset --segnet-checkpoint /scratch/<user>/segnet_out/checkpoint_0.pt
"