from models.t2t_vit import *
from utils import load_for_transfer_learning 
import torch
import warnings
import math
import time
warnings.filterwarnings('ignore')

# create model
model = t2t_vit_14(img_size=32).to('cuda')

# load the pretrained weights
load_for_transfer_learning(model,'pretrained_weights/81.5_T2T_ViT_14.pth.tar', use_ema=True, strict=False, num_classes=10)  # change num_classes based on dataset, can work for different image size as we interpolate the position embeding for different image size.


print(model)

start_time = time.time()
for _ in range(10):
    with torch.no_grad():
        random_img=torch.randn(64,3,32,32).to('cuda')
        output=model(random_img)
print("--- %s seconds ---" % (time.time() - start_time))
