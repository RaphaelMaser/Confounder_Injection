#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import pandas as pd
'''
127.0.0.1:8000:/?token=417c65a720ebe817507356246b10f9a925d3b89cbb60ed50
'''
import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sbs
import seaborn_image as sbsi
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from Framework import Models
import warnings
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, fixed
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import wandb
import ast

from ray.tune.integration.wandb import (
    WandbTrainableMixin,
    wandb_mixin,
)


warnings.filterwarnings("ignore",category=FutureWarning)


class plot:
    def __init__(self):
        self.fontsize = 20

    def accuracy_vs_epoch(self, results):
        fig, ax = plt.subplots(1,2,figsize=(8*2,4))
        fig.suptitle("Accuracy vs Epoch", fontsize=self.fontsize)
        model_name = results["model_name"][0]

        classification_accuracy = results.pivot("epoch", "confounder_strength", "classification_accuracy")
        confounder_accuracy = results.pivot("epoch", "confounder_strength", "confounder_accuracy")

        sbs.lineplot(data=classification_accuracy, ax=ax[0]).set(title=f"{model_name}\nClassification accuracy", ylim=(0.45,1.05))
        sbs.lineplot(data=confounder_accuracy, ax=ax[1]).set(title=f"{model_name}\nConfounder accuracy", ylim=(0.45,1.05))
        plt.tight_layout()

    def accuracy_vs_epoch_all(self, results):
        results_list = [d for _,d in results.groupby(["confounder_strength"])]
        fig, ax = plt.subplots(2,len(results_list), figsize=(6*len(results_list),10))
        fig.suptitle("Comparison between models", fontsize=self.fontsize)

        for i in range(0,len(results_list)):
            confounder_strength = results_list[i]["confounder_strength"].iloc[0]
            classification_accuracy = results_list[i].pivot(index="epoch", columns="model_name", values="classification_accuracy")
            confounder_accuracy = results_list[i].pivot(index="epoch", columns="model_name", values="confounder_accuracy")
            sbs.lineplot(data=classification_accuracy, ax=ax[0][i]).set(title=f"Classification Accuracy\n(confounder strength: {confounder_strength})", ylim=(0.45,1.05))
            sbs.lineplot(data=confounder_accuracy, ax=ax[1][i]).set(title=f"Confounder Accuracy\n(confounder strength: {confounder_strength})", ylim=(0.45,1.05))
        fig.tight_layout()

    def accuracy_vs_strength(self, results):
        #print(results)
        fig, ax = plt.subplots(1,1,figsize=(8,4))
        fig.suptitle("Accuracy vs Strength", fontsize=self.fontsize)
        model_name = results["model_name"][0]

        classification_accuracy = results.pivot("epoch", "confounder_strength", "classification_accuracy")
        confounder_accuracy = results.pivot("epoch", "confounder_strength", "confounder_accuracy")

        class_max_acc = classification_accuracy.max(axis='rows')
        class_mean_acc = classification_accuracy.mean(axis='rows')
        class_max_and_mean_acc = pd.concat({"max": class_max_acc, "mean": class_mean_acc}, axis='columns')
        sbs.lineplot(data=class_max_and_mean_acc, markers=True).set(title=f"{model_name}\nClassification Accuracy", ylim=(0.45,1.05))
        plt.tight_layout()
        #conf_max_acc = confounder_accuracy.max(axis='rows')
        #conf_mean_acc = confounder_accuracy.mean(axis='rows')
        #conf_max_and_mean_acc = pd.concat({"max": conf_max_acc, "mean": conf_mean_acc}, axis='columns')
        #sbs.lineplot(data=conf_max_and_mean_acc, markers=True, ax=ax[1]).set(title=f"{model_name}\nConfounder Accuracy", ylim=(0.45,1.05))


    # plot distributions (includes all samples of the class)
    def tsne(self, x, y, n):
        x = np.reshape(x, (x.shape[0],-1))
        x_embedded = TSNE(random_state=42, n_components=n, learning_rate="auto", init="pca").fit_transform(x)
        #print("t_SNE shape: ",x_embedded.shape)
        sbs.scatterplot(x=x_embedded[:,0], y=x_embedded[:,1], hue=y)
        plt.title("t-SNE", fontsize=19)
        plt.ylabel("Probability")
        plt.xlabel("Value")
        plt.show()

    def image(self, x):
        plt.imshow(x[0,:,:],cmap='gray')
        plt.colorbar()
        #plt.title("a synthetic training image");
        plt.xticks(np.arange(0), ())
        plt.yticks(np.arange(0), ())
        #plt.savefig('synthetic_sample.jpg', format='jpg', dpi=300)
        plt.show()

    def class_images(self, x, gray=True, title=None):
        images = [x[0][0], x[int(len(x)/2)+1][0]]
        #sbsi.imshow(x[0][0],ax=ax[0], gray=True, vmin=vmin, vmax=vmax).set(title="Class 0")
        #sbsi.imshow(x[int(len(x)/2)+1][0],ax=ax[1], gray=gray, vmin=vmin, vmax=vmax).set(title="Class 1")
        self.images(images, gray=True)

    def images(self, x, gray=False, title=None, model_name="None"):
        vmin = np.amin(x)
        vmax = np.amax(x)

        plots = len(x)
        fig, ax = plt.subplots(1,plots)
        fig.suptitle(f"{model_name}\n{title}", fontsize = self.fontsize)

        for i in range(plots):
            sbsi.imshow(x[i], ax=ax[i], gray=gray, vmax=vmax, vmin=vmin).set(title=f"Class {i}")
        plt.tight_layout()


    def image_slider(self, train_x):
        plt.ion()
        max = len(train_x) - 1
        interact(self.image_n, train_x=fixed(train_x), n=widgets.IntSlider(min=0, max=max, step=1, value=0));


    def image_n(self, train_x, n):
        plt.imshow(train_x[n][0])
        plt.show()
        return

    def confounding_impact(self,x):
        sbs.lineplot(x)
        return

class plot_from_csv:
    def __init__(self, fontsize=18):
        self.fontsize = fontsize
        pass

    # if None then newest is taken
    def accuracy_vs_epoch(self, project_csv_path, config_filter, groupby):
        df_list = self.convert_and_filter_df(project_csv_path, config_filter)
        title = ""
        for cf in config_filter:
            title = f"{title} \n {cf}={config_filter[cf]}"
        fig, ax = plt.subplots(1,2,figsize=(8*2,6))
        fig.suptitle("Accuracy vs Epoch", fontsize=self.fontsize)
        #model_name = df["model"]

        #for df in df_list:
            #name = df["model"][0]
        #for df in range(len(df_list)):
            #df_list[df].to_csv(f"df {df}.csv")
        #df_list[1].to_csv("df file.csv")
        df = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        #df.to_csv("df file.csv")
        classification_accuracy = df.pivot("epoch", groupby, "classification_accuracy")
        confounder_accuracy = df.pivot("epoch", groupby, "confounder_accuracy")

        sbs.lineplot(data=classification_accuracy, ax=ax[0]).set(title=f"Classification accuracy\n{title}", ylim=(0.45,1.05))
        sbs.lineplot(data=confounder_accuracy, ax=ax[1]).set(title=f"Confounder accuracy\n{title}", ylim=(0.45,1.05))

        plt.tight_layout()
        return



    def convert_and_filter_df(self, project_csv_path, config_filter):
        project_df = pd.read_json(project_csv_path)
        #project_df.to_csv("project_df.csv")
        # filter data

        # prefiltering of filters which are in config
        delete = []
        for cf in config_filter:
            config = project_df["config"]
            for line in range(len(config)):
                if cf in config[line] and config_filter[cf] != config[line][cf]:
                    delete.append(line)
        project_df = project_df.drop(index=delete).reset_index(drop=True)
        #project_df.to_csv("project_df.csv")


        merged_dfs = []
        # für jeden run einzeln
        for i in range(0,len(project_df["config"])):

            history_dict = project_df["history"][i]

            #for h in project_df["history"][i]:
            #    history_dict[h] = project_df["history"][i][h]


            config_dict = {}
            for c in project_df["config"][i]:
                config_dict[c] = {k:project_df["config"][i][c] for k in range(0,len(project_df["history"].iloc[i]["confounder_accuracy"]))}


            history_frame = pd.DataFrame.from_dict(history_dict)
            config_frame = pd.DataFrame.from_dict(config_dict)

            #history_frame.to_csv("history.csv")
            #config_frame.to_csv("config.csv")

            merged_dfs.append(pd.concat([history_frame.reset_index(drop=True), config_frame.reset_index(drop=True)], axis=1))
            #merged_dfs.append(pd.concat([history_frame, config_frame], axis=1, ignore_index=True))
            #merged_dfs.append(history_frame+config_frame)

        filtered_dfs = []
        for df in merged_dfs:
            for cf in config_filter:
                df = df[df[cf] == config_filter[cf]]
            if not df.empty:
                filtered_dfs.append(df)


        return filtered_dfs


class CfDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, domain=None, confounder=None):
        self.x = x
        self.y = y

        if domain == None:
            domain = np.zeros((len(y)))
        if confounder == None:
            confounder = np.zeros((len(y)))

        self.domain = domain
        self.confounder = confounder

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], {'y': self.y[idx], 'domain_labels': self.domain[idx], 'confounder_labels':self.confounder[idx]}

    def confounder_size(self):
        confounder_samples = self.confounder[self.confounder != -1]
        return len(confounder_samples)

class create_dataloader:
    def __init__(self, x=None, y=None, domain=None, confounder=None, batch_size=1):
        #assert(x != None and y != None)
        self.x = x
        self.y = y
        self.domain_labels = domain
        self.confounder_labels = confounder
        self.batch_size = batch_size


    # def split_dataset(self, dataset):
    #     # split dataset
    #     train_size = int(self.split * len(dataset))
    #     test_size = len(dataset) - train_size
    #     train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])
    #     return train_dataset, test_dataset


    def get_dataloader(self):
        #if len(self.x) <= 0:
        #    return None
        tensor_x = torch.Tensor(self.x)
        tensor_y = torch.Tensor(self.y).long()
        tensor_domain = torch.Tensor(self.domain_labels).long()
        tensor_confounder_labels= torch.Tensor(self.confounder_labels).long()

        dataset = CfDataset(x=tensor_x, y=tensor_y, domain=tensor_domain, confounder=tensor_confounder_labels)
        #train_dataset, test_dataset = self.split_dataset(dataset)
        # TODO delete unnecessary stuff
        # create DataLoader
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle=True)

        return train_dataloader


class generator:
    def __init__(self, mode, samples, confounding_factor, overlap=0, seed=42, params=None, de_correlate_confounder = False, domain=0, conditioning=-1):
        np.random.seed(seed)
        self.x = None
        self.y = None
        self.confounder_labels = None
        self.domain_labels = None

        self.debug = False
        self.samples = samples
        self.confounded_samples = confounding_factor
        self.overlap = overlap
        self.domain = domain
        self.de_correlate_confounder = de_correlate_confounder
        self.conditioning = conditioning

        if mode == "br-net" or mode == "br_net":
            self.br_net(params)
        elif mode == "black_n_white":
            self.black_n_white()
        elif mode == "br_net_simple":
            self.br_net_simple(params)
        else:
            raise AssertionError("Generator mode not valid")

    def br_net(self, params=None):
        N = self.samples # number of subjects in a group

        self.domain_labels = np.full((N*2), self.domain)
        self.confounder_labels = np.full((N*2), -1)

        if self.samples <= 0:
            return None, None

        overlap = self.overlap
        assert(overlap % 2 == 0 and overlap <= 32)
        overhang = int(overlap / 2)

        if params is None:
            params = [
                [[1, 4], [2, 6]], # real feature
                [[5, 4], [10, 6]] # confounder_labels
            ]

        confounded_samples = int(N*self.confounded_samples)

        if self.de_correlate_confounder == True:
            # 2 confounding effects between 2 groups, confounder_labels chosen by chance
            self.confounder_labels[:confounded_samples] = np.random.randint(0,2,size=confounded_samples)
            self.confounder_labels[N:N + confounded_samples] = np.random.randint(0,2, size=confounded_samples)
            #print("de-correlated: ",self.confounder_labels)

        else:
            # 2 confounding effects between 2 groups
            self.confounder_labels[:confounded_samples] = 0
            self.confounder_labels[N:N + confounded_samples] = 1
            #print("correlated: ", self.confounder_labels)

        if self.conditioning != -1:
            self.confounder_labels[self.conditioning*N:(self.conditioning+1)*N] = np.full((N),-1)


        cf = np.zeros((N*2))
        for i in range(0,len(self.confounder_labels)):
            index = self.confounder_labels[i]
            if index != -1:
                #print("Index is ",index)
                cf[i] = np.random.uniform(params[1][index][0], params[1][index][1])
            else:
                cf[i] = np.nan


        # 2 major effects between 2 groups
        mf = np.zeros((N*2))
        mf[:N] = np.random.uniform(params[0][0][0], params[0][0][1],size=N)
        mf[N:] = np.random.uniform(params[0][1][0], params[0][1][1],size=N)

        # simulate images
        x = np.zeros((N*2,1,32,32))
        y = np.zeros((N*2))
        y[N:] = 1

        for i in range(N*2):
            # add real feature
            x[i,0,:16 + overhang, :16 + overhang] += self.gkern(kernlen=16+overhang, nsig=5) * mf[i]
            x[i,0, 16 - overhang:, 16 - overhang:] += self.gkern(kernlen=16+overhang, nsig=5) * mf[i]

            # check if confounder_labels should be added
            if not np.isnan(cf[i]):
                x[i,0, 16 - overhang:, :16 + overhang] += self.gkern(kernlen=16+overhang, nsig=5) * cf[i]
                x[i,0,:16 + overhang,16 - overhang:] += self.gkern(kernlen=16+overhang, nsig=5) * cf[i]

            # add random noise
            x[i] = x[i] + np.random.normal(0,0.01,size=(1,32,32))

        if self.debug:
            print("--- generator ---")
            print("Confounding factor:",self.confounded_samples)
            print("Number of samples per group")
            print("Confounded samples per group (estimate):", confounded_samples)
        self.x = x
        self.y = y


    def br_net_simple(self, params=None):
        if params is None:
            params = [
                [[1, 4], [2, 6]], # real feature
                [[5, 4], [10, 6]] # confounder_labels
            ]

        N = self.samples # number of subjects in a group
        confounded_samples = int(N*self.confounded_samples)
        labels = np.zeros((N*2,))
        labels[N:] = 1

        # 2 confounding effects between 2 groups
        cf = np.zeros((N*2,))
        cf[:N] = np.random.uniform(params[1][0][0], params[1][0][1],size=N)
        cf[N:] = np.random.uniform(params[1][1][0], params[1][1][1],size=N)

        # 2 major effects between 2 groups
        mf = np.zeros((N*2,))
        mf[:N] = np.random.uniform(params[0][0][0], params[0][0][1],size=N)
        mf[N:] = np.random.uniform(params[0][1][0], params[0][1][1],size=N)

        # simulate images
        x = np.zeros((N*2,1,32,32))
        y = np.zeros((N*2))
        y[N:] = 1
        l = 0

        for i in range(N*2):
            if i/N < 1:
                x[i,0,:16,:16] = self.gkern(kernlen=16, nsig=5)*mf[i]
            if (i % N) < confounded_samples:
                if i/N < 1:
                    x[i,0,16:,:16] = self.gkern(kernlen=16, nsig=5)*cf[i]
                if i/N >= 1:
                    x[i,0,:16,16:] = self.gkern(kernlen=16, nsig=5)*cf[i]
                l+=1
            if i/N >= 1:
                x[i,0,16:,16:] = self.gkern(kernlen=16, nsig=5)*mf[i]
            x[i] = x[i] + np.random.normal(0,0.01,size=(1,32,32))

        if self.debug:
            print("--- generator ---")
            print("Confounding factor:",self.confounded_samples)
            print("Number of samples per group")
            print("Confounded samples per group (estimate):", confounded_samples)
            print("Confounded samples per group (counted)", l/2)
        self.x = x
        self.y = y
        self.cf = cf


    def black_n_white(self):
        N = self.samples
        x = np.zeros((N*2,1,32,32))
        x[N:] = 1
        y = np.zeros((N*2))
        y[N:] = 1
        self.x = x
        self.y = y

    def gkern(self, kernlen=21, nsig=3):
        import scipy.stats as st

        """Returns a 2D Gaussian kernel array."""

        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        return kernel

    def get_data(self):
        return self.x, self.y, self.domain_labels, self.confounder_labels


class train:
    def __init__(self, mode, model, train_dataloader, device, optimizer, loss_fn):
        self.model = model
        self.mode = mode
        self.train_dataloader = train_dataloader
        self.device = device
        self.loss_fn = loss_fn
        self.accuracy = []
        self.loss = []
        self.optimizer = optimizer

    def test(self, test_dataloader):
        size = len(test_dataloader.dataset)
        confounder_size = test_dataloader.dataset.confounder_size()
        num_batches = len(test_dataloader)
        self.model.eval()
        classification_accuracy, confounder_accuracy = 0, 0

        with torch.no_grad():
            for X,label in test_dataloader:
                X = X.to(self.device)
                y = label["y"].to(self.device)
                conf = label["confounder_labels"].to(self.device)

                class_pred, domain_pred = self.model(X)
                #test_loss += self.loss_fn(class_pred, label["y"]).item()
                classification_accuracy += (class_pred.argmax(1) == y).type(torch.float).sum().item()
                confounder_accuracy += (class_pred.argmax(1) == conf).type(torch.float).sum().item()
        #test_loss /= num_batches
        classification_accuracy /= size
        if confounder_size <= 0:
            confounder_accuracy = None
        else:
            confounder_accuracy /= confounder_size

        return classification_accuracy, confounder_accuracy

    def test_DANN(self):
        size = len(self.test_dataloader.dataset)
        confounder_size = self.test_dataloader.dataset.confounder_size()
        num_batches = len(self.test_dataloader)
        self.model.eval()
        classification_accuracy, confounder_accuracy = 0, 0
        with torch.no_grad():
            for X,label in self.test_dataloader:
                X = X.to(self.device)
                y = label["y"].to(self.device)
                conf = label["confounder_labels"].to(self.device)

                class_pred, domain_pred = self.model(X)
                #test_loss += self.loss_fn(class_pred, label["y"]).item()
                classification_accuracy += (class_pred.argmax(1) == y).type(torch.float).sum().item()
                confounder_accuracy += (class_pred.argmax(1) == conf).type(torch.float).sum().item()
        #test_loss /= num_batches
        classification_accuracy /= size
        if confounder_size <= 0:
            confounder_accuracy = None
        else:
            confounder_accuracy /= confounder_size

        return classification_accuracy, confounder_accuracy


    def train_normal(self):
        self.model = self.model.to(self.device)

        self.model.train()
        for batch, (X,label) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = label["y"].to(self.device)

            # Compute prediction error
            class_pred, domain_pred = self.model(X)
            loss = self.loss_fn(class_pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return

    def train_adversarial(self, mode=None, ignore=[]):
        if mode == "DANN":
            adversarial_loss = "domain_labels"
        elif mode == "CF_free":
            adversarial_loss = "confounder_labels"
        else:
            adversarial_loss = None

        self.model = self.model.to(self.device)

        # loss functions
        class_crossentropy = nn.CrossEntropyLoss()
        domain_crossentropy = nn.CrossEntropyLoss(ignore_index=-1)

        self.model.train()
        for batch, (X,label) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            y = label["y"].to(self.device)
            adv = label[adversarial_loss].to(self.device)

            # prediction
            class_pred, conf_pred = self.model(X)

            # compute error
            class_loss = class_crossentropy(class_pred, y)
            domain_loss = domain_crossentropy(conf_pred, adv)
            loss = class_loss + domain_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
        return

    def run(self):
        if isinstance(self.model, Models.SimpleConv_DANN) or isinstance(self.model, Models.Br_Net_DANN):
            self.train_adversarial(mode="DANN")

        elif isinstance(self.model, Models.SimpleConv_CF_free) or isinstance(self.model, Models.Br_Net_CF_free):
            self.train_adversarial(mode="CF_free")
        else:
            self.train_normal()

        #accuracy, loss = self.test()
        return


class confounder:
    all_results = pd.DataFrame()
    t = None
    def __init__(self, seed=42, mode="NeuralNetwork", debug=False, clean_results=False, start_timer=False, tune=False, name=None):
        torch.backends.cudnn.benchmark = True
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.mode = mode
        self.test_dataloader = []
        self.train_dataloader = None

        self.train_x = None
        self.train_y = None
        self.train_domain_labels = None
        self.train_confounder_labels = None

        self.test_x = None
        self.test_y = None
        self.test_domain_labels = None
        self.test_confounder_labels = None

        self.results = None
        self.loss = []
        self.debug = debug
        self.index = []
        self.fontsize = 18
        self.model_title_add = ""
        self.tune = tune
        self.name = name


        if start_timer:
            confounder.t = time.time()

        if clean_results:
            confounder.all_results= pd.DataFrame()

        if debug:
            print("--- constructor ---")
            #print("Model:\n",model)


    def generate_data(self, mode=None, overlap=0, samples=512, target_domain_samples=0, target_domain_confounding=0, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=False, de_correlate_confounder_target=False, conditioning=-1, params=None):
        iterations = len(test_confounding)
        self.train_x, self.test_x = np.empty((iterations,samples*2 + target_domain_samples*2,1,32,32)), np.empty((iterations,samples*2,1,32,32))
        self.train_y, self.test_y = np.empty((iterations,samples*2 + target_domain_samples*2)), np.empty((iterations,samples*2))
        self.train_domain_labels = np.empty((iterations,samples*2 + target_domain_samples*2))
        self.test_domain_labels = np.empty((iterations,samples*2))
        self.train_confounder_labels = np.empty((iterations,samples*2 + target_domain_samples*2))
        self.test_confounder_labels = np.empty((iterations,samples*2))
        self.conditioning = conditioning
        self.samples = samples
        self.target_domain_samples = target_domain_samples
        self.overlap = overlap
        self.train_confounding = train_confounding
        self.test_confounding = test_confounding
        self.target_domain_confounding = target_domain_confounding
        self.de_correlate_confounder_test = de_correlate_confounder_test
        self.de_correlate_confounder_target =de_correlate_confounder_target
        self.params = params


        if conditioning != -1:
            self.model_title_add = f"(conditioning={conditioning})"

        self.index = test_confounding

        # train data
        g_train = generator(mode=mode, samples=samples, overlap=overlap, confounding_factor=train_confounding, params=params, domain=0)
        g_train_data = g_train.get_data()
        self.train_x[0,:samples*2] = g_train_data[0]
        self.train_y[0,:samples*2] = g_train_data[1]
        self.train_domain_labels[0,:samples*2] = g_train_data[2]
        self.train_confounder_labels[0,:samples*2] = g_train_data[3]

        # append target domain_labels data to source domain_labels data
        if target_domain_samples != 0:
            g_target_domain = generator(mode=mode, samples=target_domain_samples, overlap=overlap, confounding_factor=target_domain_confounding, params=params, domain=1, de_correlate_confounder=de_correlate_confounder_target, conditioning=conditioning)
            g_target_domain_data =g_target_domain.get_data()
            self.train_x[0,samples*2:] = g_target_domain_data[0]
            self.train_y[0,samples*2:] = g_target_domain_data[1]
            self.train_domain_labels[0,samples*2:] = g_target_domain_data[2]
            self.train_confounder_labels[0,samples*2:] = g_target_domain_data[3]

        i = 0
        for cf_var in test_confounding:
            # test data
            g_test = generator(mode=mode, samples=samples, overlap=overlap, confounding_factor=cf_var, params=params, domain=0, de_correlate_confounder=de_correlate_confounder_test)
            g_test_data =g_test.get_data()
            self.test_x[i] = g_test_data[0]
            self.test_y[i] = g_test_data[1]
            self.test_domain_labels[i] = g_test_data[2]
            self.test_confounder_labels[i] = g_test_data[3]


            i += 1

            if self.debug:
                print("--- generate_data ---")
                #print("Generated Data of dimension ", self.train_x.shape)
        return self.train_x, self.train_y, self.test_x, self.test_y


    def train(self, model=Models.NeuralNetwork(32 * 32), epochs=1, device = "cuda", optimizer = None, loss_fn = nn.CrossEntropyLoss(), batch_size=1, hyper_params=None, wandb_init=None):
        name = model.get_name()

        if self.conditioning != -1:
            name += f" {self.conditioning}"

        if device == "cuda":
            if torch.cuda.is_available():
                print("CUDA detected")
            else:
                device="cpu"

        if wandb_init != None:
            if "project" not in wandb_init:
                wandb_init["project"] = "None"
            if "group" not in wandb_init:
                wandb_init["group"] = "None"
            if "time" not in wandb_init:
                wandb_init["time"] = "None"

        config = {
            "model":name,
            "epochs":epochs,
            "device": device,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "batch_size": batch_size,
            "alpha": model.alpha,
            "lr": hyper_params["lr"],
            "weight_decay": hyper_params["weight_decay"],
            "samples": self.samples,
            "target_domain_samples": self.target_domain_samples,
            "overlap": self.overlap,
            "train_confounding": self.train_confounding,
            "test_confounding": self.test_confounding,
            "target_domain_confounding": self.target_domain_confounding,
            "de_correlate_confounder_test": self.de_correlate_confounder_test,
            "de_correlate_confounder_target": self.de_correlate_confounder_target,
            "params": self.params,
        }

        if wandb_init != None:
            config["date"] = wandb_init["time"]
            wandb.init(name=name, entity="confounder_in_ml", config=config, project=wandb_init["project"], group=wandb_init["group"])

        delta_t = time.time()
        set = 0
        results = {"confounder_strength":[],"model_name":[],"epoch":[],"classification_accuracy":[], "confounder_accuracy":[]}


        if hyper_params == None:
            raise AssertionError("Choose some hyperparameter for the optimizer")
        if not 'weight_decay' in hyper_params:
            hyper_params['weight_decay'] = 0

        self.model = copy.deepcopy(model)
        model_optimizer = optimizer(params=self.model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params["weight_decay"])
        self.train_dataloader = create_dataloader(self.train_x[set],self.train_y[set], domain=self.train_domain_labels[set], confounder=self.train_confounder_labels[set], batch_size=batch_size).get_dataloader()
        for cf_var in range(0,len(self.index)):
            self.test_dataloader.append(create_dataloader(self.test_x[cf_var],self.test_y[cf_var], domain=self.test_domain_labels[cf_var], confounder=self.test_confounder_labels[cf_var], batch_size=batch_size).get_dataloader())

        for i in range(0, epochs):
            # load new results

            t = train(self.mode, self.model, self.train_dataloader, device, model_optimizer, loss_fn)
            t.run()

            for cf_var in range(0,len(self.index)):
                classification_accuracy, confounder_accuracy = t.test(self.test_dataloader[cf_var])

                results["confounder_strength"].append(self.index[cf_var])
                results["model_name"].append(self.model.get_name()+self.model_title_add)
                results["epoch"].append(i+1)
                results["classification_accuracy"].append(classification_accuracy)
                results["confounder_accuracy"].append(confounder_accuracy)

                if wandb_init != None:
                    wandb.log({"classification_accuracy":classification_accuracy, "confounder_accuracy":confounder_accuracy, "confounder_strength":self.index[cf_var], "epoch":i+1})

                # register accuracy in tune
                if self.tune:
                    assert(len(self.index==1))
                    tune.report(mean_accuracy=classification_accuracy)

        if wandb_init != None:
            wandb.config.update({"trained_model": self.model},allow_val_change=True)
            wandb.finish()
        self.results = pd.DataFrame(results)
        confounder.all_results = pd.concat([confounder.all_results, self.results], ignore_index=True)
        #confounder.all_results.append(self.results)

        if self.debug:
            print("--- train ---")
            print("Training took ",time.time() - delta_t, "s")

        return self.results

    @wandb_mixin
    def train_tune(self, config):
        #model=Models.NeuralNetwork(32 * 32), epochs=1, device = "cuda", optimizer = None, loss_fn = nn.CrossEntropyLoss(), batch_size=1, hyper_params=None):
        assert(len(self.index)==1)
        if config["device"]=="cuda":
            if not torch.cuda.is_available():
                device="cpu"
            else:
                print("CUDA detected")

        self.model = copy.deepcopy(config["model"])

        if "alpha" in config.keys():
            try:
                self.model.alpha = config["alpha"]
            except:
                pass

        model_optimizer = config["optimizer"](params=self.model.parameters(), lr=config['lr'], weight_decay=config["weight_decay"])
        self.train_dataloader = create_dataloader(self.train_x[0],self.train_y[0], domain=self.train_domain_labels[0], confounder=self.train_confounder_labels[0], batch_size=config["batch_size"]).get_dataloader()
        self.test_dataloader.append(create_dataloader(self.test_x[0],self.test_y[0], domain=self.test_domain_labels[0], confounder=self.test_confounder_labels[0], batch_size=config["batch_size"]).get_dataloader())

        for i in range(0, config["epochs"]):
            # load new results

            t = train(self.mode, self.model, self.train_dataloader, device, model_optimizer, config["loss_fn"])
            t.run()

            classification_accuracy, confounder_accuracy = t.test(self.test_dataloader[0])

            tune.report(mean_accuracy=classification_accuracy)
            wandb.log({"classification_accuracy": classification_accuracy, "confounder_accuracy": confounder_accuracy})
        return


    def plot(self, accuracy_vs_epoch=False, accuracy_vs_strength=False, tsne=False, image=False, train_images=False, test_images=False, test_image_iteration=[0], saliency=False, saliency_sample=0, smoothgrad=False, saliency_iteration=[0], image_slider=None, plot_all=False):
        p = plot()
        model_name = self.model.get_name()

        if accuracy_vs_epoch:
            p.accuracy_vs_epoch(self.results)

        if accuracy_vs_strength:
            p.accuracy_vs_strength(self.results)

        if plot_all:
            p.accuracy_vs_epoch_all(confounder.all_results)

        if tsne:
            if len(self.train_x) > 1:
                print("There are multiple arrays of data. Only showing the first one.")
            p.tsne(self.train_x[0], self.train_y[0], 2)

        if image:
            if len(self.train_x) > 1:
                print("There are multiple arrays of data. Only showing the first one.")
            p.image(self.train_x[0][0])

        if image_slider != None:
            p.image_slider(self.train_x[image_slider])

        if train_images:
            for i in test_image_iteration:
                x = self.train_x[i]
                p.images([x[0][0], x[int(len(x)/2)+1][0]], gray=True, title=f"Train-images (iteration {i})", model_name=model_name)

        if test_images:
            for i in test_image_iteration:
                x = self.test_x[i]
                p.images([x[0][0], x[int(len(x)/2)+1][0]], gray=True, title=f"Test-images (iteration {i})", model_name=model_name)

        if saliency:
            for i in saliency_iteration:
                saliency_class_0 = self.saliency_map(saliency_class=0, saliency_sample=saliency_sample, saliency_iteration=i)
                saliency_class_1 = self.saliency_map(saliency_class=1, saliency_sample=saliency_sample, saliency_iteration=i)

                p.images([saliency_class_0, saliency_class_1], title=f"Saliency map\nstrength={self.index[i]}", model_name=model_name)

        if smoothgrad:
            for i in saliency_iteration:
                saliency_class_0 = self.smoothgrad(saliency_class=0, saliency_sample=saliency_sample, saliency_iteration=i)
                saliency_class_1 = self.smoothgrad(saliency_class=1, saliency_sample=saliency_sample, saliency_iteration=i)

                p.images([saliency_class_0, saliency_class_1], title=f"SmoothGrad\nstrength={self.index[i]}", model_name=model_name)

    def smoothgrad(self, saliency_class=0, saliency_sample=0, saliency_iteration=None):
        N = 100
        noise = 0.20

        # getting the input image
        classes = len(np.unique(self.test_y))
        samples_per_class = int(self.test_x.shape[1]/classes)
        sample = saliency_class*samples_per_class + saliency_sample
        x = self.test_x[saliency_iteration][sample][0]

        # compute min and max of values, compute sigma
        x_min = np.min(x)
        x_max = np.max(x)
        sigma = noise*(x_max - x_min)

        # computing saliency
        saliency_map = np.zeros((N,32,32))

        for i in range(0,N):
            image_noise = np.random.normal(0,sigma, size=(32,32))
            x_noisy = x + image_noise
            x_noisy = torch.tensor(x_noisy, dtype=torch.float)
            x_noisy = torch.reshape(x_noisy, (1,1,32,32))
            saliency_map[i] = self.compute_saliency(x_noisy)

        saliency_map = (1/N) * np.sum(saliency_map, axis=0)
        return np.array(saliency_map)

    def saliency_map(self, saliency_class=0, saliency_sample=0, saliency_iteration=0):

        # getting the input image
        classes = len(np.unique(self.test_y))
        samples_per_class = int(self.test_x.shape[1]/classes)
        sample = saliency_class*samples_per_class + saliency_sample
        x = self.test_x[saliency_iteration][sample]
        x = torch.tensor(x, dtype=torch.float)
        x = torch.reshape(x, (1,1,32,32))

        saliency_map = self.compute_saliency(x)

        return np.array(saliency_map)

    def compute_saliency(self,x):
        self.model.eval()

        # gradients need to be computed for the image
        x.requires_grad = True

        # predict labels
        class_pred, domain_pred = self.model(x)

        # take argmax because we are only interested in the most probable class (and why the models decides in favor of it)
        # argmax returns vector with index of the maximum value (zero for class zero, one for class one)
        pred_idx = class_pred.argmax()
        class_pred[0,pred_idx].backward()
        saliency = torch.abs(x.grad)
        #print(saliency.shape)
        return saliency[0][0]

    def show_time(self):
        t = time.time() - confounder.t
        print(f"Computation took {int(t/60)} min and {int(t%60)} s")

def sync_wandb_data(project=None):
    t = time.time()
    assert(project is not None)
    entity = "confounder_in_ml"
    if project==None:
        project = "BrNet on br_net data"

    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    history_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        history_list.append(run.history(samples=100000))

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
    history_dict = [hl.to_dict() for hl in history_list]

    #print(history_dict)
    #print("\n\n")
    #print(history_dict)
    runs_df = pd.DataFrame({
        "history": history_dict,
        "config": config_list,
        "name": name_list
    })

    runs_df.to_csv(f"{project}.csv")
    #runs_df.to_pickle(f"{project}.pkl")
    runs_df.to_json(f"{project}.json")
    print(f"Syncing took {time.time() - t} seconds")

def get_dates(file):
    df = pd.read_json(file)
    df = {d["date"] for d in df["config"]}
    df = [d for d in df]
    df.sort(reverse=True)
    return df