import torch.nn.functional as F
import torch
from torch import nn

class NeuralNetwork3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32)  
        self.relu = nn.ReLU()   # funcion de activacion ReLU
        self.fc2 = nn.Linear(32, 10)     # capa de salida

    def forward(self, x):
        x = x.view(x.size(0), -1)    
        x = self.fc1(x)
        x = self.relu(x)       
        x = self.fc2(x)
        return x