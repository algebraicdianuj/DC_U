import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ResNet50s(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50s, self).__init__()
        self.backbone = resnet50(weights=None)
        
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.backbone.maxpool = nn.Identity()
        
        in_features = self.backbone.fc.in_features
        self.class_head = nn.Linear(in_features, num_classes)
        del self.backbone.fc


    def embed(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x) 

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

    def classifier(self, features):
        return self.class_head(features) 

    def forward(self, x):
        feats = self.embed(x)
        logits = self.classifier(feats)
        return logits