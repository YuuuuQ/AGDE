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
from datasets import choose_dataset
from models import create_model

def load_results(filename):
    with open(filename, "rb") as f:
        results = joblib.load(f)
    return results

def save_results(results, filename):
    with open(filename, "wb") as f:
        joblib.dump(results, f)


population_size = 5
better_ratio = 0.8
num_epoch = 100
F1 = F2 = 0.001
batchsize = 128
learning_rate = 0.01
learning_rate_agde = 0.001

LR = [0.001, 0.0001, 0.01]
F = [0.001, 0.01]

name = "CIFAR10"
# , "CIFAR100", STL10
Model = "resnet18"
#, densenet121, mobilenetv2

train_dataset, test_dataset, train_loader, test_loader, num_class = choose_dataset(name, batchsize)

for F1 in F:
    F2 = F1
    for learning_rate_agde in LR:
        population = [create_model(Model, num_class) .cuda() for _ in range(population_size)]

        criterion = torch.nn.CrossEntropyLoss()

        figure_save_path = "./Results/" + Model + "/" + name + "/" + str(learning_rate_agde) + "lr-" + str(F1) + "F/"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)


        #==========AGDE================
        print("=======Running AGDE=======")

        optimizers_agde = [optim.Adam(model.parameters(), lr=learning_rate_agde) for model in population]
        agde = AGDE(train_loader, test_loader, criterion, population_size, better_ratio, num_epoch, F1, F2)

        start_time_agde = time.time()
        agde.AGDE(optimizers_agde, population, figure_save_path + "AGDE.pkl")
        end_time_agde = time.time()
        total_time_agde = end_time_agde - start_time_agde


        agde_results = load_results(figure_save_path + "AGDE.pkl")
        agde_results["total_time"] = total_time_agde
        save_results(agde_results, figure_save_path + "AGDE.pkl")

        #==========MSGD================
        print("=======Running MSGD=======")
        optimizers_msgd = [optim.SGD(model.parameters(), lr=learning_rate) for model in population]
        msgd = GDs(num_epoch, criterion, train_loader, test_loader, population_size)

        start_time_msgd = time.time()
        msgd.GD(optimizers_msgd, population, figure_save_path + "MSGD.pkl")
        end_time_msgd = time.time()
        total_time_msgd = end_time_msgd - start_time_msgd

        msgd_results = load_results(figure_save_path + "MSGD.pkl")
        msgd_results["total_time"] = total_time_msgd
        save_results(msgd_results, figure_save_path + "MSGD.pkl")

        #==========MAdam================
        print("=======Running MAdam=======")
        optimizers_adam = [optim.Adam(model.parameters(), lr=learning_rate) for model in population]
        madam = GDs(num_epoch, criterion, train_loader, test_loader, population_size)

        start_time_madam = time.time()
        madam.GD(optimizers_adam, population, figure_save_path + "MAdam.pkl")
        end_time_madam = time.time()
        total_time_madam = end_time_madam - start_time_madam

        madam_results = load_results(figure_save_path + "MAdam.pkl")
        madam_results["total_time"] = total_time_madam
        save_results(madam_results, figure_save_path + "MAdam.pkl")

        #==========MRmsprop================
        print("=======Running MRMSprop=======")
        optimizers_rmsprop = [optim.RMSprop(model.parameters(), lr=learning_rate) for model in population]
        mrmsprop = GDs(num_epoch, criterion, train_loader, test_loader, population_size)

        start_time_mrmsprop = time.time()
        mrmsprop.GD(optimizers_rmsprop, population, figure_save_path + "MRMSprop.pkl")
        end_time_mrmsprop = time.time()
        total_time_mrmsprop = end_time_mrmsprop - start_time_mrmsprop


        mrmsprop_results = load_results(figure_save_path + "MRMSprop.pkl")
        mrmsprop_results["total_time"] = total_time_mrmsprop
        save_results(mrmsprop_results, figure_save_path + "MRMSprop.pkl")

        #==========MAdamW================
        print("=======Running MAdamW=======")
        optimizers_adamw = [optim.AdamW(model.parameters(), lr=learning_rate) for model in population]
        madamw = GDs(num_epoch, criterion, train_loader, test_loader, population_size)

        start_time_madamw = time.time()
        madamw.GD(optimizers_adamw, population, figure_save_path + "MAdamW.pkl")
        end_time_madamw = time.time()
        total_time_madamw = end_time_madamw - start_time_madamw

        madamw_results = load_results(figure_save_path + "MAdamW.pkl")
        madamw_results["total_time"] = total_time_madamw
        save_results(madamw_results, figure_save_path + "MAdamW.pkl")

        #==========MRAdam================
        print("=======Running MRAdam=======")
        optimizers_radam = [optim.RAdam(model.parameters(), lr=learning_rate) for model in population]
        mradam = GDs(num_epoch, criterion, train_loader, test_loader, population_size)

        start_time_mradam = time.time()
        mradam.GD(optimizers_radam, population, figure_save_path + "MRAdam.pkl")
        end_time_mradam = time.time()
        total_time_mradam = end_time_mradam - start_time_mradam

        mradam_results = load_results(figure_save_path + "MRAdam.pkl")
        mradam_results["total_time"] = total_time_mradam
        save_results(mradam_results, figure_save_path + "MRAdam.pkl")

        #==========MNAdam================
        print("=======Running MNAdam=======")
        optimizers_nadam = [optim.NAdam(model.parameters(), lr=learning_rate) for model in population]
        mnadam = GDs(num_epoch, criterion, train_loader, test_loader, population_size)

        start_time_mnadam = time.time()
        mnadam.GD(optimizers_nadam, population, figure_save_path + "MNAdam.pkl")
        end_time_mnadam = time.time()
        total_time_mnadam = end_time_mnadam - start_time_mnadam

        mnadam_results = load_results(figure_save_path + "MNAdam.pkl")
        mnadam_results["total_time"] = total_time_mnadam
        save_results(mnadam_results, figure_save_path + "MNAdam.pkl")


plot_results(agde_results, msgd_results, madam_results, mrmsprop_results, madamw_results,
             mradam_results, mnadam_results, figure_save_path)
plot_running_times(figure_save_path)
plot_individual_results(agde_results, figure_save_path)

save_to_excel(figure_save_path)
