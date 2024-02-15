import torch
from torch import nn
import torch.nn.functional as F

class SentimentAnalysis(nn.Module):
    def __init__(self, 
                 input_shape,
                 hidden_units,
                 output):
        super().__init__()
        self.layer_1 = nn.Linear(input_shape, hidden_units)
        self.layer_2 = nn.ReLU()
        self.layer_3 = nn.Embedding(hidden_units, output)
        
    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        y = self.layer_3(y)
        y = F.softmax(y)
        
        return y

