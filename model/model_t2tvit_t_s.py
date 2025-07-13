from model.T2T_ViT.models.t2t_vit import *
from model.T2T_ViT.utils import load_for_transfer_learning 
import torch.nn as nn
import time


class T2TVisionTransformerT_S(nn.Module):
    def __init__(self, num_classes=10):
        super(T2TVisionTransformerT_S, self).__init__()
        model = t2t_vit_14(img_size=32).to('cuda')
        # load the pretrained weights
        load_for_transfer_learning(model,'pretrained_weights/81.7_T2T_ViTt_14.pth.tar', use_ema=True, strict=False, num_classes=num_classes)
        self.vit = model 

    def embed(self, x):
        return self.vit.features(x)

    def classifier(self, features):
        return self.vit.classifier(features)

    def forward(self, x):
        features = self.embed(x)
        logits = self.classifier(features)
        return logits
