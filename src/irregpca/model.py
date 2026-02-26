import torch
import torch
from torch import nn

class DefaultModel(nn.Module):
    def __init__(self, d: int = 1, width = 20):
        super().__init__()
        self.layer1 = nn.Linear(d, width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, 1)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x
    
def train_only(j, models):
    for i, model in enumerate(models):
        for parameters in model.parameters():
            parameters.requires_grad = (i == j)