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
import time

params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder_labels
]

e = datetime.datetime.now()
epochs = 10000
samples = 128
cpus_per_trial = 128
ray.init(num_cpus=128)

search_space = {
    "epochs":epochs,
    "batch_size": tune.choice([64,128,256]),
    "optimizer":torch.optim.Adam,
    "alpha":tune.uniform(0,1),
    "alpha2":tune.uniform(0,1),
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
    if not "wandb_init" in config:
        config["wandb_init"] = None

    c = CI.confounder()
    c.generate_data(mode="br_net", samples=512, target_domain_samples=target_domain_samples, target_domain_confounding=target_domain_confounding, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=de_correlate_confounder_target, de_correlate_confounder_test=de_correlate_confounder_test, params=params)
    c.train(use_tune=True, use_wandb=True, epochs=config["epochs"], model = config["model"], optimizer=config["optimizer"], hyper_params={"batch_size": config["batch_size"],"lr": config["lr"], "weight_decay": config["weight_decay"]}, wandb_init=config["wandb_init"])


def run_tune():
    #reporter = CLIReporter(max_progress_rows=1, max_report_frequency=120)
    scheduler = tune.schedulers.PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations=
        {
            "lr":search_space["lr"],
            "weight_decay": search_space["weight_decay"],
            "batch_size": search_space["batch_size"],
            "alpha": search_space["alpha"],
            "alpha2": search_space["alpha2"],
        }
    )
    model_name = search_space["model"].get_name()
    tune.run(train_tune,num_samples=samples, config=search_space, keep_checkpoints_num=4,
             #resources_per_trial={"cpu":cpus_per_trial, "gpu":0},
             local_dir=f"~/ray_results/target_domain_samples={target_domain_samples},test_confounding={test_confounding},model={model_name}/{args.date}")


def BrNet_hyperparams():
    search_space["model"] = Models.BrNet()
    search_space["wandb_init"]["group"] = "BrNet"
    run_tune()

def BrNet_CF_free_labels_entropy_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_entropy(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_entropy"
    run_tune()

def BrNet_CF_free_labels_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_entropy(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_entropy_conditioned"
    run_tune()

def BrNet_CF_free_labels_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_corr"
    run_tune()

def BrNet_CF_free_labels_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_corr_conditioned"
    run_tune()

def BrNet_CF_free_features_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_features_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_features_corr"
    run_tune()

def BrNet_CF_free_features_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_features_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_features_corr_conditioned"
    run_tune()

def BrNet_DANN_entropy_hyperparams():
    search_space["model"] = Models.BrNet_DANN_entropy(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_DANN_entropy"
    run_tune()

def BrNet_DANN_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_DANN_entropy(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_DANN_entropy_conditioned"
    run_tune()

def BrNet_DANN_corr_hyperparams():
    search_space["model"] = Models.BrNet_DANN_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_DANN_corr"
    run_tune()

def BrNet_DANN_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_DANN_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_DANN_corr_conditioned"
    run_tune()

def BrNet_CF_free_DANN_labels_entropy_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy"
    run_tune()

def BrNet_CF_free_DANN_labels_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_conditioned"
    run_tune()

def BrNet_CF_free_DANN_labels_entropy_features_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_features_corr"
    run_tune()

def BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned"
    run_tune()

os.environ['WANDB_MODE'] = 'run'
#os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('-d', action="store", dest="date", help="Define the date")
parser.add_argument('-c', action="store", dest="c", help="Define the cpu count")
parser.add_argument('-test_confounding', type=int, action="store", dest="test_confounding", help="Define strength of confounder in test data")
parser.add_argument('-target_domain_samples', type=int, action="store", dest="target_domain_samples", help="Define number of target domain samples")
parser.add_argument('-target_domain_confounding', type=int, action="store", dest="target_domain_confounding", help="Define confounding of target domain")
parser.add_argument('-de_correlate_confounder_target', type=int, action="store", dest="de_correlate_confounder_target", help="Define if target domain should be de-correlated")
parser.add_argument('-de_correlate_confounder_test', type=int, action="store", dest="de_correlate_confounder_test", help="Define if target domain should be de-correlated")
args = parser.parse_args()
search_space["wandb_init"]["batch_date"] = args.date
test_confounding = args.test_confounding
target_domain_samples = args.target_domain_samples
target_domain_confounding = args.target_domain_confounding
de_correlate_confounder_target = args.de_correlate_confounder_target
de_correlate_confounder_test = args.de_correlate_confounder_test

# run experiments
BrNet_hyperparams()
BrNet_CF_free_labels_entropy_hyperparams()
BrNet_CF_free_labels_entropy_conditioned_hyperparams()
BrNet_CF_free_labels_corr_hyperparams()
BrNet_CF_free_labels_corr_conditioned_hyperparams()
BrNet_CF_free_features_corr_hyperparams()
BrNet_CF_free_features_corr_conditioned_hyperparams()
BrNet_DANN_entropy_hyperparams()
BrNet_DANN_entropy_conditioned_hyperparams()
BrNet_DANN_corr_hyperparams()
BrNet_DANN_corr_conditioned_hyperparams()
BrNet_CF_free_DANN_labels_entropy_hyperparams()
BrNet_CF_free_DANN_labels_entropy_conditioned_hyperparams()
BrNet_CF_free_DANN_labels_entropy_features_corr_hyperparams()
BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned_hyperparams()

# for i in range(0,20):
#     print(f"Waited for {i} minutes")
#     time.sleep(60)