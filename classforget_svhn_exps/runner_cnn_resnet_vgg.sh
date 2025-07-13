#!/bin/bash

python resnet_pre_procedure.py
python resnet_unlearn_relative_methods.py
python vgg_pre_procedure.py
python vgg_unlearn_relative_methods.py