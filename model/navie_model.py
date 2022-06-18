import os
import torch
from torch import nn


class MlpSdcNet(nn.Module):
    def __init__(self, num_future_states):
        super(MlpSdcNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, X):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits