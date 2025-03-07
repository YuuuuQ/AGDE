import torch
import torch.optim as optim
import random
import copy
import time
from train_test import test
import numpy as np
import joblib

class AGDE:
    def __init__(self, train_loader, test_loader, criterion,
                 population_size, better_ratio, num_epoch, F1, F2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.population_size = population_size
        self.num_epoch = num_epoch
        self.better_ratio = better_ratio
        self.F1 = F1
        self.F2 = F2
        self.t = 0
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

    def train_better_individual(self, model, optimizer, best_model_deepcopy):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            for param, best_param in zip(model.parameters(), best_model_deepcopy.parameters()):
                param.data = param.data + self.F1 * (best_param.data - param.data)
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
        avg_loss = train_loss / len(self.train_loader)
        return avg_loss, accuracy

    def train_bad_individual(self, model, optimizer, best_model_deepcopy, populations):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            j, k = random.sample(range(self.population_size), 2)
            for param, best_param, param_j, param_k in zip(model.parameters(), best_model_deepcopy.parameters(),
                                                           populations[j].parameters(), populations[k].parameters()):
                param.data = param.data + self.F2 * (best_param.data - param.data) + self.F2 * (
                            param_j.data - param_k.data)
            optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        avg_loss = train_loss / len(self.train_loader)
        return avg_loss, accuracy

    def train_individual(self, model, optimizer, i, best_model_deepcopy, populations, population_labels):
        if population_labels[i] == 0:
            avg_loss, accuracy = self.train(model, optimizer, self.train_loader)
        elif population_labels[i] == 1:
            avg_loss, accuracy = self.train_better_individual(model, optimizer, best_model_deepcopy)
        else:
            avg_loss, accuracy = self.train_bad_individual(model, optimizer, best_model_deepcopy, populations)
        return avg_loss, accuracy

    def generate_list(self, n, m):
        result = [2] * n
        result[0] = 0
        ones_count = int(m * n - 1)
        ones_indices = random.sample(range(1, n), ones_count)
        for index in ones_indices:
            result[index] = 1
        return result


    def AGDE(self, optimizers, population, filename):
        best_model_deepcopy = copy.deepcopy(population[0])
        population_labels = self.generate_list(self.population_size, self.better_ratio)

        for epoch in range(self.num_epoch):
            print(population_labels)
            print(f"Epoch [{epoch + 1}/{self.num_epoch}]")
            epoch_test_accuracies = []
            epoch_test_losses = []
            epoch_train_accuracies = []
            epoch_train_losses = []
            epoch_times = []

            for i, model in enumerate(population):
                start_time = time.time()
                optimizer = optimizers[i]
                if epoch == 0:
                    train_loss, train_acc = self.train(model, optimizer, self.train_loader)
                else:
                    train_loss, train_acc = self.train_individual(model, optimizer, i, best_model_deepcopy,
                                                                   population,
                                                                   population_labels)

                test_loss, test_acc = test(model, self.test_loader, self.criterion, self.device)
                end_time = time.time()
                elapsed_time = end_time - start_time

                epoch_test_accuracies.append(test_acc)
                epoch_test_losses.append(test_loss)
                epoch_train_accuracies.append(train_acc)
                epoch_train_losses.append(train_loss)
                epoch_times.append(elapsed_time)

                print(
                    f"Population {i + 1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%| Time: {elapsed_time:.2f}s")

            self.population_test_accuracies.append(epoch_test_accuracies)
            self.population_test_losses.append(epoch_test_losses)
            self.population_train_accuracies.append(epoch_train_accuracies)
            self.population_train_losses.append(epoch_train_losses)
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



