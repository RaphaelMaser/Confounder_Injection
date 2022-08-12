#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Framework.Confounder_Injection as CI
import Framework.Models as Models
import importlib
importlib.reload(Models)
importlib.reload(CI)
import torch
import time
import datetime
import ray
import argparse
import os

# In[2]:


params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder_labels
    ]

os.environ['WANDB_MODE'] = 'run'
epochs = 10000
e = datetime.datetime.now()
t = f"{e.year}.{e.month}.{e.day} {e.hour}:{e.minute}:{e.second}"

# In "Validation of synthetic data" the case were the training_data was completely confounded was tested to show that neural networks indeed fit to confounding factors in images. Now the hypothesis is that if we have a small set of unconfounded data we can either use a confounder_labels-free neural network or a DANN to unlearn the confounders. For establishing a performance baseline we need to test the SimpleConv on a dataset consisting of the confounded set and the small unconfounded set, otherwise the conditions would not be equal for the different networks.

# # No confounders in target and test_data

# In this case there are 512 samples from the source domain_labels (with correlating confounders) and a varying number of samples (16 or 64) from the target domain_labels (with no confounders).

# In this case the network is able to achieve the highest possible accuracy. When the confounder_labels is absent the network seems to be able to use the real features for distinguishing the classes and when the confounder_labels is present the network can use it to achieve higher accuracy.

# ### With 16 training-samples from target population

# In[3]:

wandb_init = {
    "project": "BrNet on br_net data",
    "date": t,
    "group": "BrNet",
}

BrNet_hyperparams = {
    "lr": 0.001695,
    "weight_decay": 0.000002422,
    "batch_size": 256,
}

BrNet_CF_free_labels_entropy_hyperparams = {
    "lr": 0.0001946,
    "weight_decay": 0.0008954,
    "batch_size": 256,
    "alpha": 0.9824,
}

BrNet_CF_free_labels_entropy_conditioned_hyperparams = {
    "lr": 0.000237,
    "weight_decay": 0.000003514,
    "batch_size": 256,
    "alpha": 0.3843,
}

BrNet_CF_free_labels_corr_hyperparams = {
    "lr": 0.0005323,
    "weight_decay": 0.0004125,
    "batch_size": 64,
    "alpha": 0.4892,
}

BrNet_CF_free_labels_corr_conditioned_hyperparams = {
    "lr": 0.0005194,
    "weight_decay": 0.000003571,
    "batch_size": 128,
    "alpha": 0.5483,
}

BrNet_CF_free_features_corr_hyperparams = {
    "lr": 0.00008257,
    "weight_decay": 0.001969,
    "batch_size": 128,
    "alpha": 0.9846,
}

BrNet_CF_free_features_corr_conditioned_hyperparams = {
    "lr": 0.00002901,
    "weight_decay": 0.00006131,
    "batch_size": 256,
    "alpha": 0.8677,
}

BrNet_DANN_entropy_hyperparams = {
    "lr": 0.0007306,
    "weight_decay": 0.000002674,
    "batch_size": 256,
    "alpha": 0.324,
}

BrNet_DANN_entropy_conditioned_hyperparams = {
    "lr": 0.0003249,
    "weight_decay": 0.000003879,
    "batch_size": 64,
    "alpha": 0.08466,
}

BrNet_DANN_corr_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

BrNet_DANN_corr_conditioned_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

BrNet_CF_free_DANN_labels_entropy_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

BrNet_CF_free_DANN_labels_entropy_conditioned_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

BrNet_CF_free_DANN_labels_entropy_features_corr_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}


def run_experiments(model, hyperparams):
    if "alpha" in hyperparams:
        model.alpha = hyperparams["alpha"]
    if "alpha2" in hyperparams:
        model.alpha2 = hyperparams["alpha2"]

    # target_domain_unconfounded_test_unconfounded_16_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=0, train_confounding=1, test_confounding=[0], params=params)
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':hyperparams["lr"], 'weight_decay':hyperparams["weight_decay"]})

    # target_domain_unconfounded_test_confounded_16_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=0, train_confounding=1, test_confounding=[1], params=params)
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':hyperparams["lr"], 'weight_decay':hyperparams["weight_decay"]})

    # target_domain_confounded_decorrelated_0_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=0, target_domain_confounding=1, de_correlate_confounder_target=True, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=True, params=params)
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':hyperparams["lr"], 'weight_decay':hyperparams["weight_decay"]})

    # target_domain_confounded_decorrelated_16_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=1, de_correlate_confounder_target=True, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=True, params=params)
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':hyperparams["lr"], 'weight_decay':hyperparams["weight_decay"]})

    # target_domain_confounded_decorrelated_128_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=128, target_domain_confounding=1, de_correlate_confounder_target=True, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=True, params=params)
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':hyperparams["lr"], 'weight_decay':hyperparams["weight_decay"]})



parser = argparse.ArgumentParser()
parser.add_argument('-i', action="store", type=int, dest="experiment_number", help="Define the number of experiment to execute")
parser.add_argument('-v', action="store", type=int, dest="experiment_number_add", help="Define the number of experiment to execute")
parser.add_argument('-d', action="store", dest="date", help="Define the date")
args = parser.parse_args()
number = args.experiment_number + 10*args.experiment_number_add
wandb_init["batch_date"] = args.date

if number == 0:
    run_experiments(Models.BrNet(), BrNet_hyperparams)
elif number == 1:
    run_experiments(Models.BrNet_CF_free_labels_entropy(alpha=None), BrNet_CF_free_labels_entropy_hyperparams)
elif number == 2:
    run_experiments(Models.BrNet_CF_free_labels_entropy(alpha=None, conditioning=0), BrNet_CF_free_labels_entropy_conditioned_hyperparams)
elif number == 3:
    run_experiments(Models.BrNet_CF_free_labels_corr(alpha=None), BrNet_CF_free_labels_corr_hyperparams)
elif number == 4:
    run_experiments(Models.BrNet_CF_free_labels_corr(alpha=None, conditioning=0), BrNet_CF_free_labels_corr_conditioned_hyperparams)
elif number == 5:
    run_experiments(Models.BrNet_CF_free_features_corr(alpha=None), BrNet_CF_free_features_corr_hyperparams)
elif number == 6:
    run_experiments(Models.BrNet_CF_free_features_corr(alpha=None, conditioning=0), BrNet_CF_free_features_corr_conditioned_hyperparams)
elif number == 7:
    run_experiments(Models.BrNet_DANN_entropy(alpha=None), BrNet_DANN_entropy_hyperparams)
elif number == 8:
    run_experiments(Models.BrNet_DANN_entropy(alpha=None, conditioning=0), BrNet_DANN_entropy_conditioned_hyperparams)
elif number == 9:
    run_experiments(Models.BrNet_DANN_corr(alpha=None), BrNet_DANN_corr_hyperparams)
elif number == 10:
    run_experiments(Models.BrNet_DANN_corr(alpha=None, conditioning=0), BrNet_DANN_corr_conditioned_hyperparams)
elif number == 11:
    run_experiments(Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None), BrNet_CF_free_DANN_labels_entropy_hyperparams)
elif number == 12:
    run_experiments(Models.BrNet_CF_free_DANN_labels_entropy(alpha=None, alpha2=None, conditioning=0), BrNet_CF_free_DANN_labels_entropy_conditioned_hyperparams)
elif number == 13:
    run_experiments(Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None), BrNet_CF_free_DANN_labels_entropy_features_corr_hyperparams)
elif number == 14:
    run_experiments(Models.BrNet_CF_free_DANN_labels_entropy_features_corr(alpha=None, alpha2=None, conditioning=0), BrNet_CF_free_DANN_labels_entropy_features_corr_conditioned_hyperparams)



# In[18]:


#%%
