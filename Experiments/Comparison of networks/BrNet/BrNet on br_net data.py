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


# In[2]:


params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder_labels
    ]

epochs = 10
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
    "time": t,
    "group": "BrNet",
}

Br_Net_hyperparams = {
    "lr": 0.0003537,
    "weight_decay": 0.000001366,
    "batch_size": 128,
}

Br_Net_CF_free_labels_entropy_hyperparams = {
    "lr": 0.00008257,
    "weight_decay": 0.001969,
    "batch_size": 128,
    "alpha": 0.9846,
}

Br_Net_CF_free_labels_entropy_conditioned_hyperparams = {
    "lr": 0.00008257,
    "weight_decay": 0.001969,
    "batch_size": 128,
    "alpha": 0.9846,
}

Br_Net_CF_free_labels_corr_hyperparams = {
    "lr": 0.0005194,
    "weight_decay": 0.000003571,
    "batch_size": 128,
    "alpha": 0.5483,
}

Br_Net_CF_free_labels_corr_conditioned_hyperparams = {
    "lr": 0.0005194,
    "weight_decay": 0.000003571,
    "batch_size": 128,
    "alpha": 0.5483,
}

Br_Net_CF_free_features_corr_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

Br_Net_CF_free_features_corr_conditioned_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

Br_Net_DANN_entropy_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

Br_Net_DANN_entropy_conditioned_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

Br_Net_DANN_corr_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

Br_Net_DANN_corr_conditioned_hyperparams = {
    "lr": 0.000495,
    "weight_decay": 0.00003478,
    "batch_size": 256,
    "alpha": 0.699,
}

def run_experiments(model):

    # target_domain_unconfounded_test_unconfounded_16_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=0, train_confounding=1, test_confounding=[0], params=params)
    model_name = model.get_name() + "_hyperparams"
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=model_name["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':model_name["lr"], 'weight_decay':model_name["weight_decay"]})

    # target_domain_unconfounded_test_confounded_16_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=0, train_confounding=1, test_confounding=[1], params=params)
    model_name = model.get_name() + "_hyperparams"
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=model_name["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':model_name["lr"], 'weight_decay':model_name["weight_decay"]})

    # target_domain_confounded_decorrelated_0_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=0, target_domain_confounding=1, de_correlate_confounder_target=True, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=True, params=params)
    model_name = model.get_name() + "_hyperparams"
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=model_name["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':model_name["lr"], 'weight_decay':model_name["weight_decay"]})


    # target_domain_confounded_decorrelated_16_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=1, de_correlate_confounder_target=True, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=True, params=params)
    model_name = model.get_name() + "_hyperparams"
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=model_name["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':model_name["lr"], 'weight_decay':model_name["weight_decay"]})

    # target_domain_confounded_decorrelated_16_samples
    c = CI.confounder(clean_results=True, start_timer=True)
    c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=128, target_domain_confounding=1, de_correlate_confounder_target=True, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=True, params=params)
    model_name = model.get_name() + "_hyperparams"
    c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=model_name["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':model_name["lr"], 'weight_decay':model_name["weight_decay"]})



parser = argparse.ArgumentParser()
parser.add_argument('experiment_number', type=int, help="Define the number of experiment to execute")
parser.add_argument('date', type=str, help="Define the date")
args = parser.parse_args()
wandb_init["batch_time"] = args.date

if args.experiment_number == 0:
    run_experiments(Models.Br_Net())
elif args.experiment_number == 1:

    run_experiments(Models.Br_Net_CF_free_labels_entropy(alpha=None))
elif args.experiment_number == 2:
    run_experiments(Models.Br_Net_CF_free_labels_entropy(alpha=None, conditioning=True))
elif args.experiment_number == 3:
    run_experiments(Models.Br_Net_CF_free_labels_corr(alpha=None))
elif args.experiment_number == 4:
    run_experiments(Models.Br_Net_CF_free_labels_corr(alpha=None, conditioning=True))
elif args.experiment_number == 5:
    run_experiments(Models.Br_Net_CF_free_features_corr(alpha=None))
elif args.experiment_number == 6:
    run_experiments(Models.Br_Net_CF_free_features_corr(alpha=None, conditioning=True))
elif args.experiment_number == 7:
    run_experiments(Models.Br_Net_DANN_entropy(alpha=None))
elif args.experiment_number == 8:
    run_experiments(Models.Br_Net_DANN_entropy(alpha=None, conditioning=True))
elif args.experiment_number == 9:
    run_experiments(Models.Br_Net_DANN_corr(alpha=None))
elif args.experiment_number == 10:
    run_experiments(Models.Br_Net_DANN_corr(alpha=None, conditioning=True))



# In[18]:


#%%
