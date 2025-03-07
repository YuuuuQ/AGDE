import torch
import torch.nn as nn

def train(model, optimizer, data_loader, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / total
    avg_loss = train_loss / len(data_loader)
    return avg_loss, accuracy

def test(model, data_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(data_loader)
    return avg_loss, accuracy