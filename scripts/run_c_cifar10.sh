#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropriate environment
conda activate torcher

i=1
python cond.py --exp $i --model_name cnn_s --ipc 2 --save_img
python cond.py --exp $i --model_name cnn_s --ipc 10 --save_img

