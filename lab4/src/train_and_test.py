import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
    avg_loss = test_loss / num_batches
    acurracy = 100 * correct / size
    return avg_loss, acurracy

def train_and_evaluate(epochs, loss_fn, optimizer, model, train_dataloader, eval_dataloader, lr, DEVICE):
    results = []
    for t in range(epochs):    
           
        train_loss, train_acc = train_model(train_dataloader, model, loss_fn, optimizer, DEVICE)
        eval_loss, eval_acc = test_model(eval_dataloader, model, loss_fn, DEVICE)

        results.append((t, lr, train_loss, eval_loss, train_acc, eval_acc))
       
    df = pd.DataFrame(results, columns=['Epoch', 'Tasa de aprendizaje','Train Loss', 'Eval Loss', 'Train Accuracy', 'Eval Accuracy'])
    return df

def plot_results(result_arrays, results_name = ['']):
    colores = ['blue', 'green', 'red', 'orange']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    
    for i, results in enumerate(result_arrays):
        plt.plot(results['Epoch'], results['Train Loss'], label=f'Entrenamiento {results_name[i]}', linestyle='--', color=colores[i])
        plt.plot(results['Epoch'], results['Eval Loss'], label=f'Evaluación {results_name[i]}', color=colores[i])

    plt.xlabel('Épocas')
    plt.ylabel('Perdida')
    plt.title('Evolución de la perdida')
    plt.legend()

    plt.subplot(1, 3, 2)
    
    for i, results in enumerate(result_arrays):
        plt.plot(results['Epoch'], results['Train Accuracy'], label=f'Entrenamiento {results_name[i]}', linestyle='--', color=colores[i])
        plt.plot(results['Epoch'], results['Eval Accuracy'], label=f'Evaluación {results_name[i]}', color=colores[i])

    plt.xlabel('Épocas')
    plt.ylabel('Accuracy (%)')
    plt.title('Evolución de accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

