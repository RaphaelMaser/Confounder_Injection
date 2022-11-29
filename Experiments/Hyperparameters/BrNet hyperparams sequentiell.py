import sys
from ray.tune.integration.wandb import wandb_mixin
import Framework.Confounder_Injection as CI
import Framework.Models as Models
import importlib
import torch
from ray import tune
from ray import air
import ray
from ray.tune import CLIReporter
import datetime
import argparse
import os
import numpy as np

params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder_labels
]

e = datetime.datetime.now()
cpus_per_trial = 1
ray.init()
local_dir = os.path.join("/dev/shm","ray_results")

os.path.join(local_dir, f"{np.random.randint(sys.maxsize)}")

# Define the search space for ray
search_space = {
    "batch_size": tune.choice([64,128,256]),
    "optimizer":torch.optim.Adam,
    "alpha":1,
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
        "dir": local_dir,
    },
}

# Parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', action="store", dest="date", help="Define the date")
parser.add_argument('-c', action="store", dest="c", help="Define the cpu count")
parser.add_argument('-pbt', action="store", type=int, dest="pbt", help="Define the scheduler")
parser.add_argument('-epochs', action="store", type=int, dest="epochs", help="Define the scheduler")
parser.add_argument('-samples', action="store", type=int, dest="samples", help="Define the scheduler")
parser.add_argument('-finetuning', action="store", type=int, dest="finetuning", help="Enable finetuning?")
parser.add_argument('-test_samples', action="store", type=int, dest="test_samples", help="Define the scheduler")
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
finetuning = args.finetuning
test_samples = args.test_samples

# Training procedure which will be run in parallel by Ray during PBT
def train_tune(config, checkpoint_dir=None):
    if "alpha" in config:
        config["model"].alpha = config["alpha"]
    if "alpha2" in config:
        config["model"].alpha2 = config["alpha2"]
    if not "wandb_init" in config:
        config["wandb_init"] = None

    # pre-train model on the confounded dataset and then finetune it on the small dataset
    if finetuning:
        c_ft = CI.confounder()
        c_ft.generate_data(mode="br_net", samples=512, target_domain_samples=0, test_samples=test_samples, target_domain_confounding=target_domain_confounding, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=de_correlate_confounder_target, de_correlate_confounder_test=de_correlate_confounder_test, params=params)
        c_ft.train(use_tune=False, use_wandb=False, epochs=int(config["epochs"]/2), model = config["model"], optimizer=config["optimizer"], hyper_params={"batch_size": config["batch_size"],"lr": config["lr"], "weight_decay": config["weight_decay"]}, wandb_init=config["wandb_init"])

        c = CI.confounder()
        c.generate_data(mode="br_net", samples=0, target_domain_samples=target_domain_samples, test_samples=test_samples, target_domain_confounding=target_domain_confounding, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=de_correlate_confounder_target, de_correlate_confounder_test=de_correlate_confounder_test, params=params)
        c.train(use_tune=True, use_wandb=True, epochs=int(config["epochs"]/2), model = c_ft.model, optimizer=config["optimizer"], hyper_params={"batch_size": config["batch_size"],"lr": config["lr"], "weight_decay": config["weight_decay"]}, wandb_init=config["wandb_init"], checkpoint_dir=checkpoint_dir)

    # standard routine
    else:
        c = CI.confounder()
        c.generate_data(mode="br_net", samples=512, target_domain_samples=target_domain_samples, test_samples=test_samples, target_domain_confounding=target_domain_confounding, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=de_correlate_confounder_target, de_correlate_confounder_test=de_correlate_confounder_test, params=params)
        c.train(use_tune=True, use_wandb=True, epochs=config["epochs"], model = config["model"], optimizer=config["optimizer"], hyper_params={"batch_size": config["batch_size"],"lr": config["lr"], "weight_decay": config["weight_decay"]}, wandb_init=config["wandb_init"], checkpoint_dir=checkpoint_dir)


# This function executes PBT
def run_tune(search_space):
    reporter = CLIReporter(max_progress_rows=1, max_report_frequency=600*3)

    if pbt:
        scheduler = tune.schedulers.PopulationBasedTraining(
            time_attr="epoch",
            perturbation_interval=int(epochs/20),
            hyperparam_mutations=
            {
                "lr":search_space["lr"],
                "weight_decay": search_space["weight_decay"],
                "batch_size": [64,128,256],
            },
            metric="mean_accuracy",
            mode="max",
            synch=False
        )
    else:
        scheduler = None

    tuner = tune.Tuner(train_tune,
                       tune_config=tune.TuneConfig(
                           num_samples = samples,
                           scheduler=scheduler,
                       ),
                       run_config=air.RunConfig(
                           local_dir=local_dir,
                           sync_config=ray.tune.SyncConfig(syncer=None),
                           checkpoint_config=air.CheckpointConfig(
                               num_to_keep = 10,
                           ),
                       ),
                        param_space=search_space,
                       )
    tuner.fit()

    name = search_space["model"].get_name()
    print(f"----- finished ----\n"
          f"{name}\n"
          f"----- finished ----\n")


# These functions create the model indicated in the title and start PBT with the function above
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

# Disable unnecessary ray logging
os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = "1"

# run experiments
BrNet_hyperparams()
BrNet_CF_free_labels_entropy_hyperparams()
BrNet_CF_free_labels_entropy_conditioned_hyperparams()

BrNet_CF_free_features_corr_hyperparams()
BrNet_CF_free_features_corr_conditioned_hyperparams()
BrNet_DANN_entropy_hyperparams()
BrNet_DANN_entropy_conditioned_hyperparams()