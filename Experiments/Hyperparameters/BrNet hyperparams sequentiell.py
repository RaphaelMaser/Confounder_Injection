import matplotlib.pyplot as plt
from ray.tune.integration.wandb import wandb_mixin

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
import shutil

params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder_labels
]

e = datetime.datetime.now()
#epochs = 1000
cpus_per_trial = 32
#max_concurrent_trials = 32
ray.init(num_cpus=128)

search_space = {
    #"epochs":epochs,
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
        "group": "BrNet",
        "date": [f"{e.year}.{e.month}.{e.day} {e.hour}:{e.minute}:{e.second}"],
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('-d', action="store", dest="date", help="Define the date")
parser.add_argument('-c', action="store", dest="c", help="Define the cpu count")
parser.add_argument('-pbt', action="store", type=int, dest="pbt", help="Define the scheduler")
parser.add_argument('-epochs', action="store", type=int, dest="epochs", help="Define the scheduler")
parser.add_argument('-samples', action="store", type=int, dest="samples", help="Define the scheduler")
parser.add_argument('-finetuning', action="store", type=int, dest="finetuning", help="Enable finetuning?")
parser.add_argument('-test_confounding', type=int, action="store", dest="test_confounding", help="Define strength of confounder in test data")
parser.add_argument('-target_domain_samples', type=int, action="store", dest="target_domain_samples", help="Define number of target domain samples")
parser.add_argument('-target_domain_confounding', type=int, action="store", dest="target_domain_confounding", help="Define confounding of target domain")
parser.add_argument('-de_correlate_confounder_target', type=int, action="store", dest="de_correlate_confounder_target", help="Define if target domain should be de-correlated")
parser.add_argument('-de_correlate_confounder_test', type=int, action="store", dest="de_correlate_confounder_test", help="Define if target domain should be de-correlated")
args = parser.parse_args()
search_space["wandb_init"]["batch_date"] = args.date
search_space["wandb_init"]["pbt"] = args.pbt
search_space["epochs"] = args.epochs
search_space["wandb_init"]["finetuning"] = args.finetuning
test_confounding = args.test_confounding
target_domain_samples = args.target_domain_samples
target_domain_confounding = args.target_domain_confounding
de_correlate_confounder_target = args.de_correlate_confounder_target
de_correlate_confounder_test = args.de_correlate_confounder_test
samples = args.samples
pbt = args.pbt
epochs = args.epochs
#TODO just for testing
finetuning = args.finetuning

# if finetuning==1:
#     search_space["batch_size"] = target_domain_samples

#@wandb_mixin
def train_tune(config, checkpoint_dir=None):
    #print(f"--- RESSOURCES ---\n{ray.cluster_resources()}")
    if "alpha" in config:
        config["model"].alpha = config["alpha"]
    if "alpha2" in config:
        config["model"].alpha2 = config["alpha2"]
    if not "wandb_init" in config:
        config["wandb_init"] = None
    # print("Alpha: ", config["model"].alpha)
    # print("Alpha2: ", config["model"].alpha2)

    # pre-train model on the confounded dataset and then finetune it on the small dataset
    if finetuning:
        c_ft = CI.confounder()
        c_ft.generate_data(mode="br_net", samples=512, target_domain_samples=0, target_domain_confounding=target_domain_confounding, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=de_correlate_confounder_target, de_correlate_confounder_test=de_correlate_confounder_test, params=params)
        c_ft.train(use_tune=False, use_wandb=False, epochs=int(config["epochs"]/2), model = config["model"], optimizer=config["optimizer"], hyper_params={"batch_size": config["batch_size"],"lr": config["lr"], "weight_decay": config["weight_decay"]}, wandb_init=config["wandb_init"])

        c = CI.confounder()
        c.generate_data(mode="br_net", samples=0, target_domain_samples=target_domain_samples, target_domain_confounding=target_domain_confounding, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=de_correlate_confounder_target, de_correlate_confounder_test=de_correlate_confounder_test, params=params)
        c.train(use_tune=True, use_wandb=True, epochs=int(config["epochs"]/2), model = c_ft.model, optimizer=config["optimizer"], hyper_params={"batch_size": config["batch_size"],"lr": config["lr"], "weight_decay": config["weight_decay"]}, wandb_init=config["wandb_init"])

    # standard routine
    else:
        c = CI.confounder()
        c.generate_data(mode="br_net", samples=512, target_domain_samples=target_domain_samples, target_domain_confounding=target_domain_confounding, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=de_correlate_confounder_target, de_correlate_confounder_test=de_correlate_confounder_test, params=params)
        c.train(use_tune=True, use_wandb=True, epochs=config["epochs"], model = config["model"], optimizer=config["optimizer"], hyper_params={"batch_size": config["batch_size"],"lr": config["lr"], "weight_decay": config["weight_decay"]}, wandb_init=config["wandb_init"])


def run_tune(search_space):
    reporter = CLIReporter(max_progress_rows=1, max_report_frequency=600*3)
    if pbt:
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
            },
            metric="mean_accuracy",
            mode="max"
        )
    else:
        scheduler = None

    local_dir = "/mnt/lscratch/users/rmaser/ray_results"
    tune.run(train_tune, num_samples=samples, config=search_space, keep_checkpoints_num=4, progress_reporter=reporter, scheduler=scheduler,
             resources_per_trial={"cpu":cpus_per_trial, "gpu":0},
             #max_concurrent_trials=max_concurrent_trials,
             local_dir=local_dir
    )
    # remove ray_results folder
    #time.sleep(20)
    #shutil.rmtree(local_dir, ignore_errors=True)



def BrNet_hyperparams():
    search_space["model"] = Models.BrNet()
    search_space["wandb_init"]["group"] = "BrNet"
    run_tune(search_space)

def BrNet_CF_free_labels_entropy_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_entropy(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_entropy"
    run_tune(search_space)

def BrNet_CF_free_labels_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_entropy(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_entropy_conditioned"
    run_tune(search_space)

def BrNet_CF_free_labels_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_corr"
    run_tune(search_space)

def BrNet_CF_free_labels_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_labels_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_labels_corr_conditioned"
    run_tune(search_space)

def BrNet_CF_free_features_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_features_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_features_corr"
    run_tune(search_space)

def BrNet_CF_free_features_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_features_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_features_corr_conditioned"
    run_tune(search_space)

def BrNet_DANN_entropy_hyperparams():
    search_space["model"] = Models.BrNet_DANN_entropy(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_DANN_entropy"
    run_tune(search_space)

def BrNet_DANN_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_DANN_entropy(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_DANN_entropy_conditioned"
    run_tune(search_space)

def BrNet_DANN_corr_hyperparams():
    search_space["model"] = Models.BrNet_DANN_corr(alpha=None)
    search_space["wandb_init"]["group"] = "BrNet_DANN_corr"
    run_tune(search_space)

def BrNet_DANN_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_DANN_corr(alpha=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_DANN_corr_conditioned"
    run_tune(search_space)

def BrNet_CF_free_DANN_labels_entropy_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy"
    run_tune(search_space)

def BrNet_CF_free_DANN_labels_entropy_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_conditioned"
    run_tune(search_space)

def BrNet_CF_free_DANN_labels_entropy_features_corr_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_features_corr"
    run_tune(search_space)

def BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned_hyperparams():
    search_space["model"] = Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None, conditioning=0)
    search_space["wandb_init"]["group"] = "BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned"
    run_tune(search_space)

os.environ['WANDB_MODE'] = 'run'
#os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = "0"

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

for i in range(0,5):
    print(f"Waited for {i} minutes")
    time.sleep(60)

