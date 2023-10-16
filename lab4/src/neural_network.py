from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32)  # capa oculta
        self.sigmoid = nn.Sigmoid()  # funcion de activacion
        self.fc2 = nn.Linear(32, 10)  # capa de salida

    def forward(self, x):
        x = x.view(x.size(0), -1)  # aplanar
        x = self.fc1(x) 
        x = self.sigmoid(x)  # aplicar la funcion sigmoide
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Convertir la salida en una distribución de probabilidad
        return x

  