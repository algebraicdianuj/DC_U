#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropriate environment
conda activate torcher

python multiround_unlearn.py --exp 1 --model_name resnet_s --retrain_lr 1e-1 --dataset cinic10 --batch_size 64 --unlearning_rounds 3