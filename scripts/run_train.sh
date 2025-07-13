#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropriate environment
conda activate torcher

for i in {1..3}
do
    python train.py --exp $i --model_name resnet_s --batch_size 64 --lr 1e-1 --momentum 0.9 --weight_decay 5e-4 --epochs 200 --warmup 10
    python train.py --exp $i --model_name resnetlarge_s --batch_size 64 --lr 1e-1 --momentum 0.9 --weight_decay 5e-4 --epochs 200 --warmup 10
    # python train.py --exp $i --model_name cnn_s --batch_size 64 --lr 1e-1 --momentum 0.9 --weight_decay 5e-4 --epochs 200 --warmup 10
done
