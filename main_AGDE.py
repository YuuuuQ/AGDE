import torch
import torch.nn as nn
from GD_models import GDs
from AGDE import AGDE
import torch.optim as optim
from train_test import *
from models import LeNet5
import os
import joblib
import time
from draw import *
from datasets import choose_dataset, choose_dataset_subset_mnist
from models import create_model

def load_results(filename):
    with open(filename, "rb") as f:
        results = joblib.load(f)
    return results

def save_results(results, filename):
    with open(filename, "wb") as f:
        joblib.dump(results, f)


population_size = 5
# better_ratio = 0.8
num_epoch = 100
# F1 = F2 = 0.001
batchsize = 128
# learning_rate_agde = 0.001
learning_rate = 0.001

LR = [0.001, 0.01]
F = [[0.001, 0.0001], [0.001, 0.001], [0.01, 0.001], [0.01, 0.01]]
better_ratios = [0.8, 0.6]

name = "CIFAR10"
# , "CIFAR100", STL10
Model = "resnet18"
#, densenet121, mobilenetv2
# train_dataset, test_dataset, train_loader, test_loader, num_class = choose_dataset_subset_mnist(batchsize)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

train_dataset, test_dataset, train_loader, test_loader, num_class = choose_dataset(name, batchsize)


results_list = []

for better_ratio in better_ratios:
        for F_M in F:
            F1, F2 = F_M[0], F_M[1]
            for learning_rate_agde in LR:

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                population = [create_model(Model, num_class).to(device) for _ in range(population_size)]

                criterion = torch.nn.CrossEntropyLoss()

                figure_save_path = f"./Results_AGDE/{Model}/{name}/{learning_rate_agde}lr-{F_M}F-{better_ratio}br/"
                if not os.path.exists(figure_save_path):
                    os.makedirs(figure_save_path)

                # ==========AGDE================
                print(
                    f"=======Running AGDE with lr={learning_rate_agde}, F={F1}, br={better_ratio}=======")

                optimizers_agde = [optim.Adam(model.parameters(), lr=learning_rate_agde, weight_decay=1e-4) for model in
                                   population]
                agde = AGDE(train_loader, test_loader, criterion, population_size, better_ratio, num_epoch, F1, F2)

                start_time_agde = time.time()
                agde.AGDE(optimizers_agde, population, figure_save_path + "AGDE.pkl")
                end_time_agde = time.time()
                total_time_agde = end_time_agde - start_time_agde

                agde_results = load_results(figure_save_path + "AGDE.pkl")
                agde_results["total_time"] = total_time_agde
                save_results(agde_results, figure_save_path + "AGDE.pkl")


                test_loss, test_accuracy = test(population[0], test_loader, criterion, device)
                with open(figure_save_path + "results.txt", "w") as f:
                    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
                    f.write(f"Test Loss: {test_loss:.4f}\n")

                results_list.append({
                    "Learning Rate": learning_rate_agde,
                    "F1": F1,
                    "F2": F2,
                    "Better Ratio": better_ratio,
                    "Test Accuracy": test_accuracy,
                    "Test Loss": test_loss,
                    "Total Time": total_time_agde
                })


df = pd.DataFrame(results_list)
df.to_excel("./AGDE_parameter_tuning_results.xlsx", index=False)
print("Parameter tuning results saved to AGDE_parameter_tuning_results.xlsx")

