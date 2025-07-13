#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropriate environment
conda activate torcher

for i in {1..3}
do
    python ablative_v1.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset cinic10 --batch_size 64 --ipc 800 
    python ablative_v2.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset cinic10 --batch_size 64 --ipc 800 
    python ablative_v3.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset cinic10 --batch_size 64 --ipc 800 
    python ablative_v4.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset cinic10 --batch_size 64 --ipc 800 
    python ablative_v5.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset cinic10 --batch_size 64 --ipc 800 
    python ablative_v6.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset cinic10 --batch_size 64 --ipc 800  
    python ablative_v7.py --exp $i --model_name resnet_s --retrain_lr 1e-1 --unlearning_mode uniform --dataset cinic10 --batch_size 64 --ipc 800 
done