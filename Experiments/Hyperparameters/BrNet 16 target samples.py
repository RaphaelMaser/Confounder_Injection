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

params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder
]

e = datetime.datetime.now()
epochs = 10000
samples = 100
target_domain_samples = 16
max_concurrent_trials = 16

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
        "time": [f"{e.year}.{e.month}.{e.day} {e.hour}:{e.minute}:{e.second}"],
        "group": "BrNet",
    },
}


##
c = CI.confounder()
model = Models.Br_Net()
search_space["model"] = model
#search_space["wandb_init"]["group"] = "BrNet"

c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=target_domain_samples, target_domain_confounding=1, train_confounding=1, test_confounding=[1], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params)
#c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=0, target_domain_confounding=1, train_confounding=1, test_confounding=[1], params=params)

reporter = CLIReporter(max_progress_rows=1, max_report_frequency=120)
analysis = tune.run(c.train_tune,num_samples=samples, progress_reporter=reporter, config=search_space,  max_concurrent_trials=max_concurrent_trials)#, scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", max_t=epochs))
#scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", max_t=max_t),


##
c = CI.confounder()
model = Models.Br_Net_CF_free(alpha=None)
search_space["model"] = model
search_space["wandb_init"]["group"] = "BrNet CF free"
search_space["alpha"] = tune.uniform(0,1)

c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=target_domain_samples, target_domain_confounding=1, train_confounding=1, test_confounding=[1], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params)

reporter = CLIReporter(max_progress_rows=1, max_report_frequency=120)
analysis = tune.run(c.train_tune,num_samples=samples, progress_reporter=reporter, config=search_space, max_concurrent_trials=max_concurrent_trials)#, scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", max_t=epochs))


##
c = CI.confounder()
model = Models.Br_Net_CF_free(alpha=None)
search_space["model"] = model
search_space["wandb_init"]["group"] = "BrNet CF free conditioning=0"

c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=target_domain_samples, target_domain_confounding=1, train_confounding=1, test_confounding=[1], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params, conditioning=0)

reporter = CLIReporter(max_progress_rows=1, max_report_frequency=120)
analysis = tune.run(c.train_tune,num_samples=samples, progress_reporter=reporter, config=search_space, max_concurrent_trials=max_concurrent_trials)#, scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", max_t=epochs))


##
c = CI.confounder()
model = Models.Br_Net_DANN(alpha=None)
search_space["model"] = model
search_space["wandb_init"]["group"] = "BrNet DANN"

c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=target_domain_samples, target_domain_confounding=1, train_confounding=1, test_confounding=[1], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params)

reporter = CLIReporter(max_progress_rows=1, max_report_frequency=120)
analysis = tune.run(c.train_tune,num_samples=samples, progress_reporter=reporter, config=search_space, max_concurrent_trials=max_concurrent_trials)#, scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", max_t=epochs))
