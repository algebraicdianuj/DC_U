Code release for the paper ["Dataset Condensation Driven Machine Unlearning"](https://arxiv.org/abs/2402.00195).

# Setup
**Platform**: Ubuntu 22+
```bash
git clone https://github.com/algebraicdianuj/DC_U.git && cd DC_U
conda create -n DCU python=3.8.19
conda activate DCU
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -U scikit-image
conda install scikit-image
pip install adversarial-robustness-toolbox
conda install -c conda-forge opacus
pip install tensorflow-privacy
pip install timm
```


# Experiments

## Base Comparison between SOTA performance mentioned [here](#sota-unlearning-implementation-references)
### VGG-16 | CIFAR-10 (downsampled->100 images per class)

I had to downsample the dataset for the sake of NTK based scrubbing method computation on 16GB RAM, 8 GB VRAM.

```bash
cd sota_method_performance/vgg16_cifar10
chmod +x runner.sh
./runner.sh
```

### CNN | CIFAR-10 (downsampled->100 images per class)

```bash
cd sota_method_performance/cnn_cifar10
chmod +x runner.sh
./runner.sh
```


### Weak CNN | CIFAR-10 (downsampled->100 images per class)

```bash
cd sota_method_performance/weaker_cnn_cifar10
chmod +x runner.sh
./runner.sh
```


## Random Forgetting (10 Percent) (Table-1)
### MLP | CIFAR10
```bash
cd cifar10_exps/mlp
chmod +x runner.sh
./runner.sh
```

### CNN | CIFAR10
```bash
cd cifar10_exps/cnn
chmod +x runner.sh
./runner.sh
```


### RESNET18 | CIFAR10
```bash
cd cifar10_exps/resnet18
chmod +x runner.sh
./runner.sh
```

### VGG16 | CIFAR10
```bash
cd cifar10_exps/vgg16
chmod +x runner.sh
./runner.sh
```

### MLP | SVHN
```bash
cd svhn_exps/mlp
chmod +x runner.sh
./runner.sh
```

### CNN | SVHN
```bash
cd svhn_exps/cnn
chmod +x runner.sh
./runner.sh
```

### RESNET18 | SVHN
```bash
cd svhn_exps/resnet18
chmod +x runner.sh
./runner.sh
```

### VGG16 | SVHN
```bash
cd svhn_exps/vgg16
chmod +x runner.sh
./runner.sh
```


## Class-wise Forgetting

### CIFAR-10
```bash
cd classforget_cifar10_exps
chmod +x runner.sh
./runner.sh
chmod +x runner_mlp_cnn_resnet_vgg.sh
./runner_mlp_cnn_resnet_vgg.sh
```

### SVHN
```bash
cd classforget_svhn_exps
chmod +x runner.sh
./runner.sh
chmod +x runner_cnn_resnet_vgg.sh
./runner_cnn_resnet_vgg.sh
```


## Effect of Size of Rememberance Sample Dataset (Images Per Class-IPC) over Performance of Unlearning
```bash
cd ipc_exp_cifar10

# K=1
cd ipc1
chmod +x runner.sh
./runner.sh

# K=10
cd ipc10
chmod +x runner.sh
./runner.sh

# K=50
cd ipc50
chmod +x runner.sh
./runner.sh
```



## Effect of different K values over different sizes of Forget set (Dataset: CIFAR-10, Model: VGG16)
### VGG16 Experiments
```bash
cd K_evaluation

# K = 45, Forgetting percentage (out of total training dataset)=1 percent
cd cifar10_vgg16_randomforget1perc_MIcondensation_K45
chmod +x runner.sh
./runner.sh

# K = 450, Forgetting percentage = 1 percent
cd cifar10_vgg16_randomforget1perc_MIcondensation_K45
chmod +x runner.sh
./runner.sh

# K = 45, Forgetting percentage = 10 percent
cd  cifar10_vgg16_randomforget1perc_MIcondensation_K45
chmod +x runner.sh
./runner.sh

# K=450, Forgetting percentage = 10 percent
cd cifar10_vgg16_randomforget10perc_MIcondensation_K450
chmod +x runner.sh
./runner.sh

# K= 450, Forgetting percentage = 50 percent
cd cifar10_vgg16_randomforget50perc_MIcondensation_K450
chmod +x runner.sh
./runner.sh
```

### MLP Experiments
#### Using Fast Distribution Matching based Dataset Condensation (Proposed) as Base
```bash
cd K_evaluation_FDMcondensation/mlp
chmod +x runny.sh
./runny.sh

cd ..
# copy the csv files with containing name 'arbitrary_uniform' from mlp/result to random_case_plotting
# copy the csv files with containing name 'classwise' from mlp/result to classwise_plotting
```

#### Using Model Inversion based Dataset Condensation (Proposed) as Base
```bash
cd K_evaluation_MIcondensation/mlp
chmod +x runny.sh
./runny.sh

cd ..
# copy the csv files with containing name 'arbitrary_uniform' from mlp/result to random_case_plotting
# copy the csv files with containing name 'classwise' from mlp/result to classwise_plotting
```





## Why One Epoch is Sufficient for Intermediate Training
```bash
cd whysingleepoch_intermediate_cifar10_cnn
python pre_procedure.py
python post_preprocedure.py
python overture_to_proposed.py
jupyter nbconvert --to notebook --execute layer_wise_gradient.ipynb --output layer_wise_gradient.ipynb
```




## Progression of Unlearning over Multiple Rounds
```bash
cd svhn_cnn_UnlearningCycles
chmod +x runner.sh
./runner.sh
cd ..
cd plotting/unlearning cycles
jupyter nbconvert --to notebook --execute radar_plt.ipynb --output radar_plt.ipynb
```


## Unlearning over Condensed Model
```bash
cd cifar10_vgg16_Condensed_retraining
chmod +x runner.sh
./runner.sh
cd ..
mv cifar10_vgg16_Condensed_retraining/result/modular_unlearning.csv plotting/Unlearning_in_Condensation
mv cifar10_vgg16_Condensed_retraining/result/recondensation_training.csv plotting/Unlearning_in_Condensation
mv cifar10_vgg16_Condensed_retraining/result/retraining.csv plotting/Unlearning_in_Condensation
mv cifar10_vgg16_Condensed_retraining/result/fisher_forgetting_stats.csv plotting/Unlearning_in_Condensation
mv cifar10_vgg16_Condensed_retraining/result/ntk_scrubbing_stats.csv plotting/Unlearning_in_Condensation
cd plotting/Unlearning_in_Condensation
jupyter nbconvert --to notebook --execute plotter.ipynb --output plotter.ipynb
```

## Unlearning as Alternative to Differential Privacy
The repository: https://github.com/awslabs/fast-differential-privacy, was used as the differentially private optimizer for training of model.

```bash
cd cifar10_vgg16_DPCompetitor_v2
chmod +x runner.sh
./runner.sh
cd ..
cd plotting/DP_competitor
jupyter nbconvert --to notebook --execute radar_plt.ipynb --output radar_plt.ipynb
jupyter nbconvert --to notebook --execute scatter.ipynb --output scatter.ipynb
```


## Unlearning and Overfitting Metrics
```bash
cd cifar10_vit_tiny_patch16_224_UnlearningAndOverfitting_v2
chmod +x runner.sh
./runner.sh
chmod +x exp.sh
./exp.sh
cd ..
# mov the csv files from cifar10_vit_tiny_patch16_224_UnlearningAndOverfitting_v2/result to plotting/unlearning_metric_and_overfitting
jupyter nbconvert --to notebook --execute plotty.ipynb --output plotty.ipynb
```


## Dataset Condensation
```bash
cd ds_condensation_benchmarking
# DS condensation via Distribution Matching
python distribution_matching.py
# DS condensation via Gradient Matching
python gradient_matching.py
# DS condensation via Fast Distribution Matching
python mi_dataset_condensation_proposed_lite.py
# DS condensation via Model Inversion
python mi_dataset_condensation_proposed.py

cd..
# copy csv files from ds_condensation_benchmarking to plotting/dataset_condensation
jupyter nbconvert --to notebook --execute plotter.ipynb --output plotter.ipynb
```


## SOTA Unlearning Implementation References

- [Fischer Forgetting and NTK Scrubbing](https://github.com/AdityaGolatkar/SelectiveForgetting)

- [Prunning and Sparsity driven Catastrophic Forgetting](https://github.com/OPTML-Group/Unlearn-Sparse)

- [Distillation based Unlearning](https://github.com/meghdadk/SCRUB)

- [Good and Bad Teacher Distillation based Unlearning](https://github.com/vikram2000b/bad-teaching-unlearning)

## SOTA Dataset Condensation References
- [Dataset condensation with gradient matching](https://github.com/VICO-UoE/DatasetCondensation)
  
- [Dataset condensation with distribution matching](https://github.com/VICO-UoE/DatasetCondensation)
  
- [Improved distribution matching for dataset condensation](https://github.com/uitrbn/IDM)





