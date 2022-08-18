import matplotlib.pyplot as plt

import Framework.Confounder_Injection as CI
import Framework.Models as Models
import importlib
importlib.reload(Models)
importlib.reload(CI)
import torch
from ray import tune
import ray
from ray.tune import CLIReporter
import datetime
import argparse
import os

params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder_labels
]

e = datetime.datetime.now()
epochs = 10000
samples = 128
max_concurrent_trials = 8
ressources_per_trial = {"cpu":4, "gpu":0}
#ray.init(num_cpus=32)


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

def train_tune(config):
    if "alpha" in config:
        config["model"].alpha = config["alpha"]
    if "alpha2" in config:
        config["model"].alpha2 = config["alpha2"]
    if not "wandb_init" in config:
        config["wandb_init"] = None

    c = CI.confounder()
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=target_domain_samples, target_domain_confounding=1, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params)
    c.train(use_tune=True, epochs=config["epochs"], model = config["model"], optimizer=config["optimizer"], hyper_params={"batch_size": config["batch_size"],"lr": config["lr"], "weight_decay": config["weight_decay"]}, wandb_init=config["wandb_init"])
    return

def run_tune():
    #reporter = CLIReporter(max_progress_rows=1, max_report_frequency=120)
    tune.run(train_tune,num_samples=samples, config=search_space, max_concurrent_trials=max_concurrent_trials, resources_per_trial=ressources_per_trial, sync_config=tune.SyncConfig(
        syncer=None  # Disable syncing
    ))


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

os.environ['WANDB_MODE'] = 'run'
os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('-i', action="store", type=int, dest="experiment_number", help="Define the number of experiment to execute")
parser.add_argument('-v', action="store", type=int, dest="experiment_number_add", help="Define the number of experiment to execute")
parser.add_argument('-d', action="store", dest="date", help="Define the date")
parser.add_argument('-test_confounding', type=int, action="store", dest="test_confounding", help="Define strength of confounder in test data")
parser.add_argument('-target_domain_samples', type=int, action="store", dest="target_domain_samples", help="Define number of target domain samples")
args = parser.parse_args()
search_space["wandb_init"]["batch_date"] = args.date
test_confounding = args.test_confounding
target_domain_samples = args.target_domain_samples

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