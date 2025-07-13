#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropriate environment
conda activate torcher

for i in {1..3}
do
    python cond_unlearn.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset svhn --batch_size 64 --ipc 100
    python cond_unlearn.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode classwise --dataset svhn --batch_size 64 --ipc 100

    python cond_unlearn.py --exp $i --model_name resnetlarge_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset svhn --batch_size 64 --ipc 100
    python cond_unlearn.py --exp $i --model_name resnetlarge_s --retrain_lr 1e-1 --unlearning_mode classwise --dataset svhn --batch_size 64 --ipc 100
done