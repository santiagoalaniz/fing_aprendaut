import torch.nn.functional as F
import torch
from torch import nn

class NeuralNetworkWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.sigmoid1 = nn.Sigmoid()     # funcion de activacion Sigmoid
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 10)     # capa de salida

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        return x