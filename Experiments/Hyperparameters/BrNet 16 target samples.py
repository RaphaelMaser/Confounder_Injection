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
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from ray.tune import CLIReporter
import wandb
import datetime
import argparse
import os

params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder_labels
]

e = datetime.datetime.now()
epochs = 10000
samples = 1
target_domain_samples = 16
max_concurrent_trials = 64
ressources_per_trial = {"cpu":2, "gpu":0}

search_space = {
    "epochs":epochs,
    "batch_size": tune.choice([64,128,256]),
    "optimizer":torch.optim.Adam,

    "alpha":None,
    "lr": tune.loguniform(1e-5,1e-1),
    "weight_decay": tune.loguniform(1e-6,1e-1),
    "wandb" : {
        "entity": "confounder_in_ml",
        "project": "Hyperparameters",
        "group": "BrNet",
    },
    "wandb_init" : {
        "entity": "confounder_in_ml",
        "project": "Hyperparameters",
        "date": [f"{e.year}.{e.month}.{e.day} {e.hour}:{e.minute}:{e.second}"],
        "group": "BrNet",
    },
}

def run_tune():
    c = CI.confounder()
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=target_domain_samples, target_domain_confounding=1, train_confounding=1, test_confounding=[1], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params)
    reporter = CLIReporter(max_progress_rows=1, max_report_frequency=120)
    analysis = tune.run(c.train_tune,num_samples=samples, progress_reporter=reporter, config=search_space, max_concurrent_trials=max_concurrent_trials, resources_per_trial=ressources_per_trial)#, scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", max_t=epochs))


def BrNet_hyperparams():
    search_space["model"] = Models.BrNet()
    search_space["wandb_init"]["group"] = "BrNet"
    run_tune()

def BrNet_CF_free_labels_entropy_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_entropy(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_entropy"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_labels_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_entropy(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_entropy_conditioned"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_labels_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_corr"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_labels_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_corr_conditioned"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_features_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_features_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_features_corr"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_features_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_features_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_features_corr_conditioned"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_DANN_entropy_hyperparams():
    search_space["model"] = Models.BrNet_DANN_entropy(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_DANN_entropy"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_DANN_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_DANN_entropy(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_DANN_entropy_conditioned"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_DANN_corr_hyperparams():
    search_space["model"] = Models.BrNet_DANN_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_DANN_corr"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_DANN_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_DANN_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_DANN_corr_conditioned"
    search_space["alpha"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_DANN_labels_entropy_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy"
    search_space["alpha"] = tune.uniform(0,1)
    search_space["alpha2"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_DANN_labels_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_conditioned"
    search_space["alpha"] = tune.uniform(0,1)
    search_space["alpha2"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_DANN_labels_entropy_features_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_features_corr"
    search_space["alpha"] = tune.uniform(0,1)
    search_space["alpha2"] = tune.uniform(0,1)
    run_tune()

def BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned"
    search_space["alpha"] = tune.uniform(0,1)
    search_space["alpha2"] = tune.uniform(0,1)
    run_tune()

os.environ['WANDB_MODE'] = 'offline'

parser = argparse.ArgumentParser()
parser.add_argument('-i', action="store", type=int, dest="experiment_number", help="Define the number of experiment to execute")
parser.add_argument('-v', action="store", type=int, dest="experiment_number_add", help="Define the number of experiment to execute")
parser.add_argument('-d', action="store", dest="date", help="Define the date")
args = parser.parse_args()
search_space["wandb_init"]["batch_date"] = args.date

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