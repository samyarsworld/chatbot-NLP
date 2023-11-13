import torch
from torch import nn


class BotModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(BotModel, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=n_input, out_features=n_hidden),
                                     nn.ReLU(),
                                     nn.Linear(in_features=n_hidden, out_features=n_hidden),
                                     nn.ReLU(),
                                     nn.Linear(in_features=n_hidden, out_features=n_output)
                                    )
        
    def forward(self, x):
        return self.layers(x)