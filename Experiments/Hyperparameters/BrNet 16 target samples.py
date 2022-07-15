import matplotlib.pyplot as plt

import Framework.Confounder_Injection as CI
import Framework.Models as Models
import importlib
importlib.reload(Models)
importlib.reload(CI)
import torch
import argparse

# parse arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--lr", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--alpha", type=float)
arguments = parser.parse_args()

params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder
]

epochs = 100
batch_size = arguments.batch_size
hyperparams = {
    "lr": arguments.lr,
    "weight_decay": arguments.weight_decay
}
alpha = arguments.alpha
#wandb.init(name="name", entity="confounder_in_ml", project="test_project", group="hyperparameters")

c = CI.confounder()
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=1, train_confounding=1, test_confounding=[1], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params)

#wandb.init(name="name", entity="confounder_in_ml", project="test_project")
c.train(model=Models.Br_Net(), epochs=epochs, device="cpu", optimizer=torch.optim.Adam, batch_size=batch_size, hyper_params=hyperparams)
#c.train(model=Models.Br_Net_CF_free(alpha), epochs=epochs, device="cpu", optimizer=torch.optim.Adam, batch_size=batch_size, hyper_params=hyperparams)
#c.train(model=Models.Br_Net_DANN(alpha), epochs=epochs, device="cpu", optimizer=torch.optim.Adam, batch_size=batch_size, hyper_params=hyperparams)

#c = CI.confounder()
#c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=1, train_confounding=1, test_confounding=[1], de_correlate_confounder_target=True, de_correlate_confounder_test=True, params=params, conditioning=0)
#c.train(model=Models.Br_Net_CF_free(alpha), epochs=epochs, device="cpu", optimizer=torch.optim.Adam, batch_size=batch_size, hyper_params=hyperparams)

