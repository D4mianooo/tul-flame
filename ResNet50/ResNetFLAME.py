import torch
import torch.nn as nn
from torchvision import models

class ResNetFLAME(nn.Module):
    def __init__(self, out_features=156):
        super(ResNetFLAME, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        features = self.backbone(x)
        params = self.head(features)
        return params
