#!/bin/bash

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path to appropriate environment
conda activate torcher

mkdir -p ./data
cd ./data
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
tar -xzf CINIC-10.tar.gz
rm CINIC-10.tar.gz
cd ..

python get_dataset.py --choice "cinic10"
