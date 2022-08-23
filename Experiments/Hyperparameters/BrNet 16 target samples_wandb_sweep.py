import matplotlib.pyplot as plt

import Framework.Confounder_Injection as CI
import Framework.Models as Models
import importlib
importlib.reload(Models)
importlib.reload(CI)
import torch
import pandas as pd
import numpy as np
import seaborn as sbs
import matplotlib.pyplot as plt
import time
from torch import nn
import wandb
import datetime
import argparse
import os

params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder_labels
]

e = datetime.datetime.now()
samples = 128
target_domain_samples = 16

search_space = {
    "method": "random",
    "metric":{
        "name": "classification_accuracy",
        "goal": "minimize"
    },
    "parameters":{
        "batch_size": {
            "values": [64,128,256],
        },
        "lr": {
            'distribution': 'uniform',
            "min": 0.000001,
            "max": 0.1
        },
        "weight_decay": {
            'distribution': 'uniform',
            "min": 0.000001,
            "max": 0.1
        },
    },
}

train_params = {
    "epochs": 10000,
    "optimizer": torch.optim.Adam,
    "model": None,
}

wandb_init = {
    "entity": "confounder_in_ml",
    "project": "Hyperparameters WandB",
    "date": [f"{e.year}.{e.month}.{e.day} {e.hour}:{e.minute}:{e.second}"],
    "group": "BrNet",
}

def train_wandb():
    # wandb.init()
    # config = wandb.config
    # if "alpha" in config:
    #     train_params["model"].alpha = config["alpha"]
    # if "alpha2" in config:
    #     train_params["model"].alpha2 = config["alpha2"]

    c = CI.confounder()
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=target_domain_samples, target_domain_confounding=1, train_confounding=1, test_confounding=[1], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params)
    c.train(wandb_sweep=True, epochs=train_params["epochs"], model = train_params["model"], optimizer=train_params["optimizer"], wandb_init=wandb_init)


def run_wandb():
    sweep_id = wandb.sweep(search_space, entity="confounder_in_ml", project="Hyperparameters WandB")
    wandb.agent(sweep_id, function=train_wandb, count=samples)

def BrNet_hyperparams():
    train_params["model"] = Models.BrNet()
    wandb_init["group"] = "BrNet"
    run_wandb()

def BrNet_CF_free_labels_entropy_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_labels_entropy(alpha=None)
    wandb_init["group"] = "BrNet_CF_free_labels_entropy"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_labels_entropy_conditioned_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_labels_entropy(alpha=None, conditioning=0)
    wandb_init["group"] = "BrNet_CF_free_labels_entropy_conditioned"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_labels_corr_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_labels_corr(alpha=None)
    wandb_init["group"] = "BrNet_CF_free_labels_corr"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_labels_corr_conditioned_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_labels_corr(alpha=None, conditioning=0)
    wandb_init["group"] = "BrNet_CF_free_labels_corr_conditioned"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_features_corr_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_features_corr(alpha=None)
    wandb_init["group"] = "BrNet_CF_free_features_corr"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_features_corr_conditioned_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_features_corr(alpha=None, conditioning=0)
    wandb_init["group"] = "BrNet_CF_free_features_corr_conditioned"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_DANN_entropy_hyperparams():
    train_params["model"] = Models.BrNet_DANN_entropy(alpha=None)
    wandb_init["group"] = "BrNet_DANN_entropy"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_DANN_entropy_conditioned_hyperparams():
    train_params["model"] = Models.BrNet_DANN_entropy(alpha=None, conditioning=0)
    wandb_init["group"] = "BrNet_DANN_entropy_conditioned"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_DANN_corr_hyperparams():
    train_params["model"] = Models.BrNet_DANN_corr(alpha=None)
    wandb_init["group"] = "BrNet_DANN_corr"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_DANN_corr_conditioned_hyperparams():
    train_params["model"] = Models.BrNet_DANN_corr(alpha=None, conditioning=0)
    wandb_init["group"] = "BrNet_DANN_corr_conditioned"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_DANN_labels_entropy_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None)
    wandb_init["group"] = "BrNet_CF_free_DANN_labels_entropy"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    search_space["parameters"]["alpha2"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_DANN_labels_entropy_conditioned_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None, conditioning=0)
    wandb_init["group"] = "BrNet_CF_free_DANN_labels_entropy_conditioned"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    search_space["parameters"]["alpha2"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_DANN_labels_entropy_features_corr_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None)
    wandb_init["group"] = "BrNet_CF_free_DANN_labels_entropy_features_corr"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    search_space["parameters"]["alpha2"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

def BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned_hyperparams():
    train_params["model"] = Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None, conditioning=0)
    wandb_init["group"] = "BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned"
    search_space["parameters"]["alpha"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    search_space["parameters"]["alpha2"] = {"min": 0.001, "max":1, 'distribution': 'uniform',}
    run_wandb()

#os.environ['WANDB_MODE'] = 'run'
#os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('-i', action="store", type=int, dest="experiment_number", help="Define the number of experiment to execute")
parser.add_argument('-v', action="store", type=int, dest="experiment_number_add", help="Define the number of experiment to execute")
parser.add_argument('-d', action="store", dest="date", help="Define the date")
args = parser.parse_args()
wandb_init["batch_date"] = args.date

number = args.experiment_number + 10*args.experiment_number_add
if number == 0:
    BrNet_hyperparams()
elif number == 1:
    BrNet_CF_free_labels_entropy_hyperparams()
elif number == 2:
    BrNet_CF_free_labels_entropy_conditioned_hyperparams()
elif number == 3:
    BrNet_CF_free_labels_corr_hyperparams()
elif number == 4:
    BrNet_CF_free_labels_corr_conditioned_hyperparams()
elif number == 5:
    BrNet_CF_free_features_corr_hyperparams()
elif number == 6:
    BrNet_CF_free_features_corr_conditioned_hyperparams()
elif number == 7:
    BrNet_DANN_entropy_hyperparams()
elif number == 8:
    BrNet_DANN_entropy_conditioned_hyperparams()
elif number == 9:
    BrNet_DANN_corr_hyperparams()
elif number == 10:
    BrNet_DANN_corr_conditioned_hyperparams()
elif number == 11:
    BrNet_CF_free_DANN_labels_entropy_hyperparams()
elif number == 12:
    BrNet_CF_free_DANN_labels_entropy_conditioned_hyperparams()
elif number == 13:
    BrNet_CF_free_DANN_labels_entropy_features_corr_hyperparams()
elif number == 14:
    BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned_hyperparams()

# for i in range(0,20):
#     print(f"Waited for {i} minutes")
#     time.sleep(60)