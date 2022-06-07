#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import pandas as pd
'''
127.0.0.1:8000:/?token=417c65a720ebe817507356246b10f9a925d3b89cbb60ed50
'''

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import warnings
import seaborn as sbs
import seaborn_image as sbsi
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from Framework import Models



warnings.filterwarnings("ignore",category=FutureWarning)


class plot:
    def __init__(self):
        pass

    def acc_loss(self, acc, loss):
        sbs.lineplot(y=acc, x=range(1,len(acc)+1))
        plt.title("Accuracy")
        plt.ylim(0,1.1)
        plt.xlim(1,len(acc))
        plt.show()
        print("With mean accuracy=",np.mean(acc))


    # plot distributions (includes all samples of the class)
    def tsne(self, x, y, n):
        x = np.reshape(x, (x.shape[0],-1))
        x_embedded = TSNE(random_state=42, n_components=n, learning_rate="auto", init="pca").fit_transform(x)
        print("t_SNE shape: ",x_embedded.shape)
        sbs.scatterplot(x=x_embedded[:,0], y=x_embedded[:,1], hue=y)
        plt.title("t-SNE")
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

    def class_images(self, x, vmin=None, vmax=None):
        fig, ax = plt.subplots(1,2)
        fig.suptitle("Class-images")
        sbsi.imshow(x[0][0],ax=ax[0], gray=True, vmin=vmin, vmax=vmax)
        sbsi.imshow(x[int(len(x)/2)+1][0],ax=ax[1], gray=True, vmin=vmin, vmax=vmax)
        #fig.colorbar(im1, ax=ax)
        #fig.colorbar(im2, ax=ax)
        plt.show()


class CfDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class create_dataloader:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        #self.split = split
        self.batch_size = batch_size


    def get_dataset(self):

        tensor_x = torch.Tensor(self.x)
        tensor_y = torch.Tensor(self.y).long()

        dataset = CfDataset(tensor_x, tensor_y)
        return dataset


    def split_dataset(self, dataset):
        # split dataset
        train_size = int(self.split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])
        return train_dataset, test_dataset


    def get_dataloader(self):
        dataset = self.get_dataset()
        #train_dataset, test_dataset = self.split_dataset(dataset)

        # create DataLoader
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle=True)

        return train_dataloader


class generator:
    def __init__(self, mode, samples, seed=42, confounded_samples=1):
        np.random.seed(seed)
        self.x = None
        self.y = None
        self.cf = None
        self.samples = samples
        self.confounded_samples = confounded_samples

        if mode == "br-net":
            self.br_net()
        elif mode == "black_n_white":
            self.black_n_white()


    def br_net(self):
        N = self.samples # number of subjects in a group
        labels = np.zeros((N*2,))
        labels[N:] = 1

        # 2 confounding effects between 2 groups
        cf = np.zeros((N*2,))
        cf[:N] = np.random.uniform(1,4,size=N)
        cf[N:] = np.random.uniform(3,6,size=N)

        # 2 major effects between 2 groups
        mf = np.zeros((N*2,))
        mf[:N] = np.random.uniform(1,4,size=N)
        mf[N:] = np.random.uniform(3,6,size=N)

        # simulate images
        x = np.zeros((N*2,1,32,32))
        y = np.zeros((N*2))
        y[N:] = 1
        for i in range(N*2):
            x[i,0,:16,:16] = self.gkern(kernlen=16, nsig=5)*mf[i]
            x[i,0,16:,:16] = self.gkern(kernlen=16, nsig=5)*cf[i]
            x[i,0,:16,16:] = self.gkern(kernlen=16, nsig=5)*cf[i]
            x[i,0,16:,16:] = self.gkern(kernlen=16, nsig=5)*mf[i]
            x[i] = x[i] + np.random.normal(0,0.01,size=(1,32,32))
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
        return self.x, self.y


class confounder_injection:
    def __init__(self, data, params):
        self.data = np.array(data)
        self.params = params

        if "inject_noise" in self.params["confounding"]:
            self.inject_noise()
        if "add_confounder" in self.params["confounding"]:
            self.add_confounder()


    def inject_noise(self):
        '''
        Injects noise in distribution
        Parameters are noise_rate (number of samples to be manipulated) and noise_factor (impact of noise)
        '''
        noise_params = self.params["confounding"]["inject_noise"]

        for noise in noise_params:
            noise_rate = noise_params[noise]["noise_rate"]
            noise_factor = noise_params[noise]["noise_factor"]
            for c in noise_params[noise]["classes"]:
                for i in range(0,len(self.data[c])):
                    if random.uniform(0, 1) < noise_rate:
                        self.data[c][i] += random.uniform(-noise_factor,noise_factor)

        return


    def add_confounder(self):
        '''
        Injects confounder in distribution
        Parameters are distribution, mean  and std
        '''
        add_confounder_params = self.params["confounding"]["add_confounder"]

        for c in add_confounder_params:
            confounder = add_confounder_params[c]

            distribution = None
            if confounder["distribution"] == "gauss":
                distribution = np.random.normal
            assert distribution != None

            for c in confounder["classes"]:
                for i in range(0,len(self.data[c])):
                    for f in confounder["features"]:
                        self.data[c][i][f] += distribution(confounder["mean"], confounder["std"])

        return


    def get_data(self):
        return self.data


class train:
    def __init__(self, mode, model, test_dataloader, train_dataloader, device, optimizer, loss_fn):
        self.model = model
        self.mode = mode
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.device = device
        self.loss_fn = loss_fn
        self.accuracy = []
        self.loss = []
        self.optimizer = optimizer


        # if optimizer == "SGD":
        #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # elif optimizer == "Adam":
        #     self.optimizer=torch.optim.Adam(self.model.parameters())
        # else:
        #     raise TypeError("Wrong Optimizer")

    def test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, accuracy = 0, 0
        with torch.no_grad():
            for X,y in self.test_dataloader:
                X,y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        accuracy /= size

        return accuracy, test_loss


    def train_normal(self):
        self.model = self.model.to(self.device)

        self.model.train()
        for batch, (X,y) in enumerate(self.train_dataloader):
            X,y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def run(self):
        self.train_normal()
        accuracy, loss = self.test()
        return accuracy, loss


class confounder:
    def __init__(self, seed=42, model=Models.NeuralNetwork(32 * 32), mode="NeuralNetwork", debug=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = model
        self.mode = mode
        self.test_dataloader = None
        self.train_dataloader = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.accuracy = []
        self.loss = []
        self.debug = debug

        if debug:
            print("Model:\n",model)
        pass

    def generate_data(self, training_data=None, test_data=None, samples=512, train_confounding=1, test_confounding=1, split=0.8):
        train_samples = int(samples*split)
        test_samples = samples - train_samples
        g_train = generator(training_data, train_samples, train_confounding)
        self.train_x, self.train_y = g_train.get_data()
        g_test = generator(training_data, test_samples, test_confounding)
        self.test_x, self.test_y =g_test.get_data()
        if self.debug:
            print("Generated Data of dimension ", self.x.shape)
        return self.train_x, self.train_y, self.test_x, self.test_y


    def train(self, epochs=1, device ="cpu", optimizer ="SGD", loss_fn = nn.CrossEntropyLoss(), batch_size=1):

        delta_t = time.time()

        for i in range(0, epochs):
            # load new data
            self.train_dataloader = create_dataloader(self.train_x,self.train_y, batch_size).get_dataloader()
            self.test_dataloader = create_dataloader(self.test_x,self.test_y, batch_size).get_dataloader()

            if self.debug:
                dataiter = iter(self.train_dataloader)
                x, y= dataiter.next()
                print("Shape of Dataloader_train:\nx = ", x.shape,"\ny = ", y.shape)

            t = train(self.mode, self.model, self.train_dataloader, self.test_dataloader,device,optimizer,loss_fn)
            accuracy, loss = t.run()
            self.accuracy.append(accuracy)
            self.loss.append(loss)


        print("Training took ",time.time() - delta_t, "s")
        return self.accuracy, self.loss

    # implement cross validation here (call train.run() multiple times)
    def cross_validate(self):
        pass
