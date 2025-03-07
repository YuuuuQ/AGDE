import torch.optim as optim
import torch
import time
from train_test import test
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import LeNet5
import numpy as np
import joblib


class GDs:
    def __init__(self, num_epoch, criterion, train_loader, test_loader, population_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.population_size = population_size
        self.best_test_accuracies = []
        self.best_test_losses = []
        self.population_test_accuracies = []
        self.population_test_losses = []
        self.population_train_accuracies = []
        self.population_train_losses = []
        self.population_times = []

    def train(self, model, optimizer, data_loader):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        accuracy = 100. * correct / total
        avg_loss = train_loss / len(data_loader)
        return avg_loss, accuracy

    def evaluate(self, model, data_loader):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(data_loader)
        return avg_loss, accuracy

    def GD(self, optimizers, population, filename):
        for epoch in range(self.num_epoch):
            print(f"Epoch [{epoch + 1}/{self.num_epoch}]")
            epoch_test_accuracies = []
            epoch_test_losses = []
            epoch_train_accuracies = []
            epoch_train_losses = []
            epoch_times = []

            for i, model in enumerate(population):
                optimizer = optimizers[i]
                start_time = time.time()
                gd_train_loss, gd_train_acc = self.train(model, optimizer, self.train_loader)
                gd_test_loss, gd_test_acc = test(model, self.test_loader, self.criterion, self.device)
                end_time = time.time()
                elapsed_time = end_time - start_time

                epoch_train_accuracies.append(gd_train_acc)
                epoch_train_losses.append(gd_train_loss)
                epoch_test_accuracies.append(gd_test_acc)
                epoch_test_losses.append(gd_test_loss)
                epoch_times.append(elapsed_time)

                print(
                    f"Individual {i + 1}: Train Loss: {gd_train_loss:.4f} | Train Acc: {gd_train_acc:.2f}% | Test Loss: {gd_test_loss:.4f} | Test Acc: {gd_test_acc:.2f}%| Time: {elapsed_time:.2f}s")

            self.population_train_accuracies.append(epoch_train_accuracies)
            self.population_train_losses.append(epoch_train_losses)
            self.population_test_accuracies.append(epoch_test_accuracies)
            self.population_test_losses.append(epoch_test_losses)
            self.population_times.append(epoch_times)

            best_individual_index = np.argmax(self.population_test_accuracies[epoch])
            best_individual = population[best_individual_index]
            best_test_loss = self.population_test_losses[epoch][best_individual_index]
            best_test_acc = self.population_test_accuracies[epoch][best_individual_index]
            best_train_loss = self.population_train_losses[epoch][best_individual_index]
            best_train_acc = self.population_train_accuracies[epoch][best_individual_index]

            self.best_test_accuracies.append(best_test_acc)
            self.best_test_losses.append(best_test_loss)

            print(
                f"Epoch {epoch + 1} Best Individual {best_individual_index + 1}: "
                f"Train Loss: {best_train_loss:.4f} | Train Acc: {best_train_acc:.2f}% | "
                f"Test Loss: {best_test_loss:.4f} | Test Acc: {best_test_acc:.2f}%")

        results = {
            "best_test_accuracies": self.best_test_accuracies,
            "best_test_losses": self.best_test_losses,
            "population_test_accuracies": self.population_test_accuracies,
            "population_test_losses": self.population_test_losses,
            "population_train_accuracies": self.population_train_accuracies,
            "population_train_losses": self.population_train_losses,
            "population_times": self.population_times
        }
        with open(filename, "wb") as f:
            joblib.dump(results, f)
        print(f"Results saved to {filename}")



