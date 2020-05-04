import torch
import torch.nn as nn
from torchvision import models

class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(256, 256, bidirectional=True)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        # added full transformer encoder stack
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        x = x.squeeze(1)
        # back to og mrnet
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x
