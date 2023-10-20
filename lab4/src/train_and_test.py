import torch
import matplotlib.pyplot as plt
import pandas as pd

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

def train_and_evaluate(epochs, loss_fn, optimizer, model, train_dataloader, eval_dataloader, DEVICE):
    results = []
    print(model)
    for t in range(epochs):
        print(f"Epoch {t+1}:")
        
        train_loss, train_acc = train_model(train_dataloader, model, loss_fn, optimizer, DEVICE)
        eval_loss, eval_acc = test_model(eval_dataloader, model, loss_fn, DEVICE)
        
        print(f"Train loss {train_loss}, Eval loss {eval_loss}, Train accuracy {train_acc}, Eval accuracy {eval_acc}\n-------------------------------")

        results.append((t, train_loss, eval_loss, train_acc, eval_acc))
    df = pd.DataFrame(results, columns=['Epoch', 'Train Loss', 'Eval Loss', 'Train Accuracy', 'Eval Accuracy'])
    return df



def plot_results(results):
    # Plot Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results['Epoch'], results['Train Loss'], label='Train Loss', marker='o')
    plt.plot(results['Epoch'], results['Eval Loss'], label='Eval Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results['Epoch'], results['Train Accuracy'], label='Train Accuracy', marker='o')
    plt.plot(results['Epoch'], results['Eval Accuracy'], label='Eval Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Evaluation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
