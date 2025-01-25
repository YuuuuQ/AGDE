# 3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import random
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import copy


def choose_dataset(name, batch_size=128):
    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_dataset, test_dataset, train_loader, test_loader

    elif name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def MNIST_resnet18():
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def CIFAR10_resnet18():
    model = resnet18(num_classes=10)
    return model


def CIFAR100_resnet18():
    model = resnet18(num_classes=100)
    return model


class LeNet4(nn.Module):
    def __init__(self):
        super(LeNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def check_input(self, x):
        if x.shape[1:] != (1, 28, 28):
            raise ValueError("Input shape must be (batch_size, 1, 28, 28)")


def train(model, optimizer, data_loader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
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


def train_best_individual(model, optimizer, data_loader):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
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


def test(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
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


def train_better_individual(model, optimizer, data_loader, best_model_deepcopy):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        for param, best_param in zip(model.parameters(), best_model_deepcopy.parameters()):
            param.data = param.data + learning_rate1 * (best_param.data - param.data)
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    avg_loss = train_loss / len(data_loader)
    return avg_loss, accuracy


def train_bad_individual(model, optimizer, data_loader, best_model_deepcopy):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    j, k = random.sample(range(num_populations), 2)
    for param, best_param, param_j, param_k in zip(model.parameters(), best_model_deepcopy.parameters(),
                                                   populations[j].parameters(), populations[k].parameters()):
        param.data = param.data + learning_rate2 * (best_param.data - param.data) + learning_rate2 * (
                    param_j.data - param_k.data)

    accuracy = 100. * correct / total
    avg_loss = train_loss / len(data_loader)
    return avg_loss, accuracy


def train_individual(model, optimizer, data_loader, i, best_model_deepcopy):
    if population_labels[i] == 0:
        avg_loss, accuracy = train_best_individual(model, optimizer, data_loader)
    elif population_labels[i] == 1:
        avg_loss, accuracy = train_better_individual(model, optimizer, data_loader, best_model_deepcopy)
    else:
        avg_loss, accuracy = train_bad_individual(model, optimizer, data_loader, best_model_deepcopy)
    return avg_loss, accuracy


def generate_list(n, m):
    result = [2] * n
    result[0] = 0
    ones_count = int(m * n - 1)
    ones_indices = random.sample(range(1, n), ones_count)
    for index in ones_indices:
        result[index] = 1
    return result


num_populations = 5
num_epochs = 20
learning_rate1 = 0.5
learning_rate2 = 0.5
better_ratio = 0.8
population_labels = generate_list(num_populations, better_ratio)
crossover_prob = 0.5
mutation_prob = 0.1
mutation_strength = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset, test_dataset, train_loader, test_loader = choose_dataset(
    'CIFAR10')


populations = [CIFAR10_resnet18().to(device) for _ in range(num_populations)]
sgd_model = CIFAR10_resnet18().to(device)
adam_model = CIFAR10_resnet18().to(device)
rmsprop_model = CIFAR10_resnet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizers = [optim.Adam(model.parameters()) for model in populations]
sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=0.01)
adam_optimizer = optim.Adam(adam_model.parameters())
rmsprop_optimizer = optim.RMSprop(rmsprop_model.parameters())

train_losses = [[] for _ in range(num_populations + 3)]
train_accuracies = [[] for _ in range(num_populations + 3)]
test_losses = [[] for _ in range(num_populations + 3)]
test_accuracies = [[] for _ in range(num_populations + 3)]


times = {f"Population {i + 1}": [] for i in range(num_populations)}
times["SGD"] = []
times["Adam"] = []
times["RMSprop"] = []


# AGDE
prev_best_loss = float('inf')
best_model_deepcopy = copy.deepcopy(populations[0])

for epoch in range(num_epochs):
    print(population_labels)
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    best_individual_num = population_labels.index(0)
    losses = []
    accuracies = []
    for i, model in enumerate(populations):
        start_time = time.time()
        optimizer = optimizers[i]
        if epoch == 0:
            train_loss, train_acc = train(model, optimizer, train_loader)
        else:
            train_loss, train_acc = train_individual(model, optimizer, train_loader, i,
                                                     best_model_deepcopy)
        test_loss, test_acc = test(model, test_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times[f"Population {i + 1}"].append(elapsed_time)
        losses.append(train_loss)
        accuracies.append(test_acc)
        train_losses[i].append(train_loss)
        train_accuracies[i].append(train_acc)
        test_losses[i].append(test_loss)
        test_accuracies[i].append(test_acc)
        print(
            f"Population {i + 1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%| Time: {elapsed_time:.2f}s")


    # SGD
    start_time = time.time()
    sgd_train_loss, sgd_train_acc = train(sgd_model, sgd_optimizer, train_loader)
    sgd_test_loss, sgd_test_acc = test(sgd_model, test_loader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times["SGD"].append(elapsed_time)
    train_losses[-3].append(sgd_train_loss)
    train_accuracies[-3].append(sgd_train_acc)
    test_losses[-3].append(sgd_test_loss)
    test_accuracies[-3].append(sgd_test_acc)
    print(
        f"SGD: Train Loss: {sgd_train_loss:.4f} | Train Acc: {sgd_train_acc:.2f}% | Test Loss: {sgd_test_loss:.4f} | Test Acc: {sgd_test_acc:.2f}%| Time: {elapsed_time:.2f}s")

    # Adam
    start_time = time.time()
    adam_train_loss, adam_train_acc = train(adam_model, adam_optimizer, train_loader)
    adam_test_loss, adam_test_acc = test(adam_model, test_loader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times["Adam"].append(elapsed_time)
    train_losses[-2].append(adam_train_loss)
    train_accuracies[-2].append(adam_train_acc)
    test_losses[-2].append(adam_test_loss)
    test_accuracies[-2].append(adam_test_acc)
    print(
        f"Adam: Train Loss: {adam_train_loss:.4f} | Train Acc: {adam_train_acc:.2f}% | Test Loss: {adam_test_loss:.4f} | Test Acc: {adam_test_acc:.2f}%| Time: {elapsed_time:.2f}s")

    # RMSprop
    start_time = time.time()
    rmsprop_train_loss, rmsprop_train_acc = train(rmsprop_model, rmsprop_optimizer, train_loader)
    rmsprop_test_loss, rmsprop_test_acc = test(rmsprop_model, test_loader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times["RMSprop"].append(elapsed_time)
    train_losses[-1].append(rmsprop_train_loss)
    train_accuracies[-1].append(rmsprop_train_acc)
    test_losses[-1].append(rmsprop_test_loss)
    test_accuracies[-1].append(rmsprop_test_acc)
    print(
        f"RMSprop: Train Loss: {rmsprop_train_loss:.4f} | Train Acc: {rmsprop_train_acc:.2f}% | Test Loss: {rmsprop_test_loss:.4f} | Test Acc: {rmsprop_test_acc:.2f}%| Time: {elapsed_time:.2f}s")

    sorted_indices = sorted(range(num_populations), key=lambda i: losses[i])

    population_labels = [2] * len(population_labels)
    population_labels[sorted_indices[0]] = 0
    for i in range(1, int(num_populations * better_ratio)):
        population_labels[sorted_indices[i]] = 1

    best_loss = losses[sorted_indices[0]]
    best_individual_num = sorted_indices[0]
    best_model_deepcopy = copy.deepcopy(populations[sorted_indices[0]])



print("Training finished!")

avg_times = {key: sum(values) / len(values) for key, values in times.items()}
plt.figure(figsize=(10, 6))
plt.bar(range(len(avg_times)), list(avg_times.values()), align='center')
plt.xticks(range(len(avg_times)), list(avg_times.keys()), rotation='vertical')
plt.ylabel('Average Time (s)')
plt.title('Average Time per Iteration')
plt.tight_layout()
plt.show()

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 8))
for i in range(num_populations):
    plt.plot(epochs, train_losses[i], label=f'Population {i + 1}')
plt.plot(epochs, train_losses[-3], label='SGD')
plt.plot(epochs, train_losses[-2], label='Adam')
plt.plot(epochs, train_losses[-1], label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for i in range(num_populations):
    plt.plot(epochs, train_accuracies[i], label=f'Population {i + 1}')
plt.plot(epochs, train_accuracies[-3], label='SGD')
plt.plot(epochs, train_accuracies[-2], label='Adam')
plt.plot(epochs, train_accuracies[-1], label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for i in range(num_populations):
    plt.plot(epochs, test_losses[i], label=f'Population {i + 1}')
plt.plot(epochs, test_losses[-3], label='SGD')
plt.plot(epochs, test_losses[-2], label='Adam')
plt.plot(epochs, test_losses[-1], label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Testing Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for i in range(num_populations):
    plt.plot(epochs, test_accuracies[i], label=f'Population {i + 1}')
plt.plot(epochs, test_accuracies[-3], label='SGD')
plt.plot(epochs, test_accuracies[-2], label='Adam')
plt.plot(epochs, test_accuracies[-1], label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.show()
