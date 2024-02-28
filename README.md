Code release for the paper "Dataset Condensation Driven Machine Unlearning".


# Experiments

Platform: Ubuntu 22

## CIFAR-10 Experiments (Table-1)
### MLP
```bash
cd cifar10_exps/mlp
chmod +x runner.sh
./runner.sh
```

### CNN
```bash
cd cifar10_exps/cnn
chmod +x runner.sh
./runner.sh
```


### RESNET18
```bash
cd cifar10_exps/resnet18
chmod +x runner.sh
./runner.sh
```

### VGG16
```bash
cd cifar10_exps/vgg16
chmod +x runner.sh
./runner.sh
```

## SVHN Experiments (Table-1)

### MLP
```bash
cd svhn_exps/mlp
chmod +x runner.sh
./runner.sh
```

### CNN
```bash
cd svhn_exps/cnn
chmod +x runner.sh
./runner.sh
```

### RESNET18
```bash
cd svhn_exps/resnet18
chmod +x runner.sh
./runner.sh
```

### VGG16
```bash
cd svhn_exps/vgg16
chmod +x runner.sh
./runner.sh
```


## Why am I Pushing for Least Epochs on Intermediate Training
```bash
cd whysingleepoch_intermediate_cifar10_cnn
python pre_procedure.py
python post_preprocedure.py
python overture_to_proposed.py
jupyter layer_wise_gradient.ipynb
```



## Progression of Unlearning over Multiple Rounds
```bash
cd svhn_cnn_UnlearningCycles
chmod +x runner.hs
./runner.sh
cd ..
cd plotting/unlearning cycles
jupyter radar_plt.ipynb
```




