import torch.nn.functional as F
import torch
from torch import nn

class NeuralNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)  # capa oculta con 64 unidades
        self.sigmoid1 = nn.Sigmoid()     # funcion de activacion Sigmoid
        self.fc3 = nn.Linear(64, 10)     # capa de salida

    def forward(self, x):
        x = x.view(x.size(0), -1)    
        x = self.fc1(x)
        x = self.sigmoid1(x)                 
        x = self.fc3(x)
        return x