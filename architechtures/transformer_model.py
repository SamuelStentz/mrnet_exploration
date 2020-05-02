import torch
import torch.nn as nn
from torchvision import models

class TRANNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        # added full transformer encoder stack
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.squeeze(1)
        # back to og mrnet
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x
