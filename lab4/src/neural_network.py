import torch.nn.functional as F
import torch
from torch import nn

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
       # x = F.softmax(x, dim=1)  # Convertir la salida en una distribución de probabilidad
        return x

def train_model(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    total_loss = 0
    correct = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / size
    acurracy = 100 * correct / size
    return avg_loss, acurracy
   

def test_model(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_loss = test_loss / num_batches
    acurracy = 100 * correct / size
    return avg_loss, acurracy