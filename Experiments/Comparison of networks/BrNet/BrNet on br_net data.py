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


# In[2]:


params = [
    [[1, 4], [3, 6]], # real feature
    [[10, 12], [20, 22]] # confounder
    ]

epochs = 10000
e = datetime.datetime.now()
t = f"{e.year}.{e.month}.{e.day} {e.hour}:{e.minute}:{e.second}"

# In "Validation of synthetic data" the case were the training_data was completely confounded was tested to show that neural networks indeed fit to confounding factors in images. Now the hypothesis is that if we have a small set of unconfounded data we can either use a confounder-free neural network or a DANN to unlearn the confounders. For establishing a performance baseline we need to test the SimpleConv on a dataset consisting of the confounded set and the small unconfounded set, otherwise the conditions would not be equal for the different networks.

# # No confounders in target and test_data

# In this case there are 512 samples from the source domain (with correlating confounders) and a varying number of samples (16 or 64) from the target domain (with no confounders).

# In this case the network is able to achieve the highest possible accuracy. When the confounder is absent the network seems to be able to use the real features for distinguishing the classes and when the confounder is present the network can use it to achieve higher accuracy.

# ### With 16 training-samples from target population

# In[3]:

wandb_init = {
    "project": "BrNet on br_net data",
    "time": t,
    "group": "BrNet",
}

BrNet_hyperparams = {
    "lr": 0.0004997,
    "weight_decay": 0.000002459,
    "batch_size": 64,
}

BrNet_CF_free_hyperparams = {
    "lr": 0.00005954,
    "weight_decay": 0.0004026,
    "batch_size": 128,
    "alpha": 0.9134,
}

BrNet_CF_free_conditioned_hyperparams = {
    "lr": 0.001252,
    "weight_decay": 0.0004509,
    "batch_size": 256,
    "alpha": 0.7702,
}

BrNet_DANN_hyperparams = {
    "lr": 0.000843,
    "weight_decay": 0.0001522,
    "batch_size": 256,
    "alpha": 0.896,
}


#
# CONFOUNDERS IN TRAINING BUT NOT IN TEST DATA
#
c = CI.confounder(clean_results=True, start_timer=True)
model = Models.Br_Net()
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=0, train_confounding=1, test_confounding=[0,1], params=params)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_hyperparams["lr"], 'weight_decay':BrNet_hyperparams["weight_decay"]})


# In[4]:


c = CI.confounder()
model = Models.Br_Net_CF_free(alpha=0.66)
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=0, train_confounding=1, test_confounding=[0,1], params=params)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_CF_free_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_CF_free_hyperparams["lr"], 'weight_decay':BrNet_CF_free_hyperparams["weight_decay"]})


# In[5]:


c = CI.confounder()
model = Models.Br_Net_CF_free_conditioned(alpha=0.7)
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=0, train_confounding=1, test_confounding=[0,1], params=params, conditioning=0)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_CF_free_conditioned_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_CF_free_hyperparams["lr"], 'weight_decay':BrNet_CF_free_hyperparams["weight_decay"]})


# In[6]:


c = CI.confounder()
model = Models.Br_Net_DANN(alpha=0.7)
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=0, train_confounding=1, test_confounding=[0,1], params=params)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_DANN_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_DANN_hyperparams["lr"], 'weight_decay':BrNet_DANN_hyperparams["weight_decay"]})


# The accuracy with more samples is the same but the network converges faster.

# In[7]:




# # De-correlated confounders in target- and test-data

# In this case there are confounders present in the data from target domain and test-set but they are de-correlated with the real features (they are rather distributed by pure chance). This is more representative of real world examples.

# In this case the SimpleConv's accuracy diminishes when the confounding strength increases, contrary to the case before. Again the network uses the confounder as approximation, if present. But this time the confounder gives wrong clues about the classes and therefore the network's accuracy drops heavily.

# ### With 16 training-samples from target population

# In[8]:


c = CI.confounder(clean_results=True)
model = Models.Br_Net()
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=1, train_confounding=1, test_confounding=[0,1], params=params, de_correlate_confounder_test=True, de_correlate_confounder_target=True)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_hyperparams["lr"], 'weight_decay':BrNet_hyperparams["weight_decay"]})


# In[9]:


c = CI.confounder()
model = Models.Br_Net_CF_free(BrNet_CF_free_hyperparams["alpha"])
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=1, train_confounding=1, test_confounding=[0,1], params=params, de_correlate_confounder_test=True, de_correlate_confounder_target=True)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_CF_free_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_CF_free_hyperparams["lr"], 'weight_decay':BrNet_CF_free_hyperparams["weight_decay"]})


# In[10]:


c = CI.confounder()
model = Models.Br_Net_CF_free_conditioned(BrNet_CF_free_hyperparams["alpha"])
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=1, train_confounding=1, test_confounding=[0,1], params=params, de_correlate_confounder_test=True, de_correlate_confounder_target=True, conditioning=0)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_CF_free_conditioned_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_CF_free_hyperparams["lr"], 'weight_decay':BrNet_CF_free_hyperparams["weight_decay"]})


# In[11]:


c = CI.confounder()
model = Models.Br_Net_DANN(BrNet_DANN_hyperparams["alpha"])
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=16, target_domain_confounding=1, train_confounding=1, test_confounding=[0,1], params=params, de_correlate_confounder_test=True, de_correlate_confounder_target=True)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_DANN_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_DANN_hyperparams["lr"], 'weight_decay':BrNet_DANN_hyperparams["weight_decay"]})


# In[12]:




# ### With 128 training-samples from target population

# In[13]:


c = CI.confounder(clean_results=True)
model = Models.Br_Net()
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=128, target_domain_confounding=1, train_confounding=1, test_confounding=[0,1], params=params, de_correlate_confounder_test=True, de_correlate_confounder_target=True)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_hyperparams["lr"], 'weight_decay':BrNet_hyperparams["weight_decay"]})


# In[14]:


c = CI.confounder()
model = Models.Br_Net_CF_free(BrNet_CF_free_hyperparams["alpha"])
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=128, target_domain_confounding=1, train_confounding=1, test_confounding=[0,1], params=params, de_correlate_confounder_test=True, de_correlate_confounder_target=True)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_CF_free_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_CF_free_hyperparams["lr"], 'weight_decay':BrNet_CF_free_hyperparams["weight_decay"]})


# In[15]:


c = CI.confounder()
model = Models.Br_Net_CF_free_conditioned(BrNet_CF_free_hyperparams["alpha"])
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=128, target_domain_confounding=1, train_confounding=1, test_confounding=[0,1], params=params, de_correlate_confounder_test=True, de_correlate_confounder_target=True, conditioning=0)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_CF_free_conditioned_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_CF_free_hyperparams["lr"], 'weight_decay':BrNet_CF_free_hyperparams["weight_decay"]})


# In[16]:


c = CI.confounder()
model = Models.Br_Net_DANN(BrNet_DANN_hyperparams["alpha"])
c.generate_data(mode="br_net", samples=512, overlap=0, target_domain_samples=128, target_domain_confounding=1, train_confounding=1, test_confounding=[0,1], params=params, de_correlate_confounder_test=True, de_correlate_confounder_target=True)
c.train(wandb_init=wandb_init, model=model, epochs=epochs, batch_size=BrNet_DANN_hyperparams["batch_size"], optimizer=torch.optim.Adam, hyper_params={'lr':BrNet_DANN_hyperparams["lr"], 'weight_decay':BrNet_DANN_hyperparams["weight_decay"]})


# In[17]:




# In[18]:


c.show_time()

