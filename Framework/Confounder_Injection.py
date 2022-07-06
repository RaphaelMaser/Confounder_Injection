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
import warnings
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual




warnings.filterwarnings("ignore",category=FutureWarning)


class plot:
    def __init__(self):
        self.fontsize = 18
        pass

    def accuracy_vs_epoch(self, acc=None, loss=None, index=[], model_name="None"):
        fig, ax = plt.subplots(1,len(acc), figsize=[4*len(acc),3])
        for i in range(len(acc)):
            sbs.lineplot(y=acc[i], x=range(1,len(acc[i])+1), ax=ax[i]).set(title=f"{model_name}\nAccuracy vs Epoch\n strength={index[i]}", ylim=(0.45,1.05))
        #plt.xlim(1,len(acc))
        plt.show()
        #print("With mean accuracy=",np.mean(acc))


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
        plt.show()


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

    def accuracy_vs_strength(self, accuracy, model_name="None"):
        step_size = 1/(len(accuracy)-1)
        index = np.arange(0, 1.1, step_size)
        total_acc_mean, total_acc_max = [], []
        for a in accuracy:
            total_acc_mean.append(np.mean(a))
            total_acc_max.append(np.max(a))
        data = {'Mean accuracy of all epochs':total_acc_mean, 'Max accuracy of all epochs':total_acc_max}

        data_df = pd.DataFrame(data, index=index)
        sbs.lineplot(data=data_df, marker='o').set(title=f"{model_name}\nAccuracy vs Strength", ylim=(0.45,1.05))
        plt.xlabel("Strength of confounder_labels")
        plt.ylabel("Accuracy")

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


class create_dataloader:
    def __init__(self, x=None, y=None, domain=[], confounder=[], batch_size=1):
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
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle=True)

        return train_dataloader


class generator:
    def __init__(self, mode, samples, confounding_factor, overlap=0, seed=42, params=None, de_correlate_confounder = False, domain=0):
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

    def test(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, accuracy = 0, 0
        with torch.no_grad():
            for X,label in self.test_dataloader:
                #X = X.to(self.device)
                #label.to(self.device)

                class_pred, domain_pred = self.model(X)
                test_loss += self.loss_fn(class_pred, label["y"]).item()
                accuracy += (class_pred.argmax(1) == label['y']).type(torch.float).sum().item()
        test_loss /= num_batches
        accuracy /= size

        return accuracy, test_loss

    def test_DANN(self):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, accuracy = 0, 0
        with torch.no_grad():
            for X,label in self.test_dataloader:
                #X = X.to(self.device)
                #label.to(self.device)

                class_pred, domain_pred = self.model(X)
                test_loss += self.loss_fn(class_pred, label["y"]).item()
                accuracy += (class_pred.argmax(1) == label['y']).type(torch.float).sum().item()
        test_loss /= num_batches
        accuracy /= size

        return accuracy, test_loss


    def train_normal(self):
        self.model = self.model.to(self.device)

        self.model.train()
        for batch, (X,label) in enumerate(self.train_dataloader):
            #X = X.to(self.device)
            #label.to(self.device)

            # Compute prediction error
            class_pred, domain_pred = self.model(X)
            loss = self.loss_fn(class_pred, label['y'])

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
            #X = X.to(self.device)

            # prediction
            class_pred, conf_pred = self.model(X)

            # compute error
            class_loss = class_crossentropy(class_pred, label["y"])
            domain_loss = domain_crossentropy(conf_pred, label[adversarial_loss])
            loss = class_loss + domain_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
        return

    def run(self):
        if isinstance(self.model, Models.SimpleConv_DANN) or isinstance(self.model, Models.Br_Net_DANN):
            self.train_adversarial(mode="DANN")
            accuracy, loss = self.test_DANN()

        elif isinstance(self.model, Models.SimpleConv_CF_free) or isinstance(self.model, Models.Br_Net_CF_free):
            self.train_adversarial(mode="CF_free")
            accuracy, loss = self.test_DANN()
        else:
            self.train_normal()
            accuracy, loss = self.test()

        return accuracy, loss


class confounder:
    def __init__(self, seed=42, mode="NeuralNetwork", debug=False):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.mode = mode
        self.test_dataloader = None
        self.train_dataloader = None

        self.train_x = None
        self.train_y = None
        self.train_domain_labels = None
        self.train_confounder_labels = None

        self.test_x = None
        self.test_y = None
        self.test_domain_labels = None
        self.test_confounder_labels = None

        self.accuracy = []
        self.loss = []
        self.debug = debug
        self.index = []
        self.fontsize = 18

        if debug:
            print("--- constructor ---")
            #print("Model:\n",model)


    def generate_data(self, mode=None, overlap=0, samples=512, target_domain_samples=0, target_domain_confounding=0, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=False, de_correlate_confounder_target=False, params=None):
        iterations = len(test_confounding)
        self.train_x, self.test_x = np.empty((iterations,samples*2 + target_domain_samples*2,1,32,32)), np.empty((iterations,samples*2,1,32,32))
        self.train_y, self.test_y = np.empty((iterations,samples*2 + target_domain_samples*2)), np.empty((iterations,samples*2))
        self.train_domain_labels = np.empty((iterations,samples*2 + target_domain_samples*2))
        self.test_domain_labels = np.empty((iterations,samples*2))
        self.train_confounder_labels = np.empty((iterations,samples*2 + target_domain_samples*2))
        self.test_confounder_labels = np.empty((iterations,samples*2))
        #self.train_domain_labels[:,samples*2:] = np.full((target_domain_samples*2),1)
        #target_domain_x = np.empty((target_domain_samples*2,1,32,32))
        #target_domain_y = np.empty((target_domain_samples*2))

        self.index = test_confounding

        i = 0
        for cf_var in test_confounding:
            # train data
            g_train = generator(mode=mode, samples=samples, overlap=overlap, confounding_factor=train_confounding, params=params, domain=0)
            g_train_data = g_train.get_data()
            self.train_x[i,:samples*2] = g_train_data[0]
            self.train_y[i,:samples*2] = g_train_data[1]
            self.train_domain_labels[i,:samples*2] = g_train_data[2]
            self.train_confounder_labels[i,:samples*2] = g_train_data[3]

            # test data
            g_test = generator(mode=mode, samples=samples, overlap=overlap, confounding_factor=cf_var, params=params, domain=0, de_correlate_confounder=de_correlate_confounder_test)
            g_test_data =g_test.get_data()
            self.test_x[i] = g_test_data[0]
            self.test_y[i] = g_test_data[1]
            self.test_domain_labels[i] = g_test_data[2]
            self.test_confounder_labels[i] = g_test_data[3]

            # append target domain_labels data to source domain_labels data
            if target_domain_samples != 0:
                g_target_domain = generator(mode=mode, samples=target_domain_samples, overlap=overlap, confounding_factor=target_domain_confounding, params=params, domain=1, de_correlate_confounder=de_correlate_confounder_target)
                g_target_domain_data =g_target_domain.get_data()
                self.train_x[i,samples*2:] = g_target_domain_data[0]
                self.train_y[i,samples*2:] = g_target_domain_data[1]
                self.train_domain_labels[i,samples*2:] = g_target_domain_data[2]
                self.train_confounder_labels[i,samples*2:] = g_target_domain_data[3]

            i += 1

            if self.debug:
                print("--- generate_data ---")
                #print("Generated Data of dimension ", self.train_x.shape)
        return self.train_x, self.train_y, self.test_x, self.test_y


    def train(self, model=Models.NeuralNetwork(32 * 32), epochs=1, device ="cpu", optimizer = None, loss_fn = nn.CrossEntropyLoss(), batch_size=1, hyper_params=None, alpha=0.5):
        delta_t = time.time()
        set = 0

        if hyper_params == None:
            raise AssertionError("Choose some hyperparameter for the optimizer")

        for cf_var in self.index:
            self.model = copy.deepcopy(model)
            model_optimizer = optimizer(params=self.model.parameters(), lr=hyper_params['lr'])
            epoch_acc = []
            for i in range(0, epochs):
                # load new data
                self.train_dataloader = create_dataloader(self.train_x[set],self.train_y[set], domain=self.train_domain_labels[set], confounder=self.train_confounder_labels[set], batch_size=batch_size).get_dataloader()
                self.test_dataloader = create_dataloader(self.test_x[set],self.test_y[set], domain=self.test_domain_labels[set], confounder=self.test_confounder_labels[set], batch_size=batch_size).get_dataloader()
                t = train(self.mode, self.model, self.test_dataloader, self.train_dataloader,device,model_optimizer,loss_fn)
                accuracy, loss = t.run()
                epoch_acc.append(accuracy)
            self.accuracy.append(epoch_acc)
            set += 1

        if self.debug:
            print("--- train ---")
            print("Training took ",time.time() - delta_t, "s")
        return self.accuracy, self.loss


    def plot(self, accuracy_vs_epoch=False, accuracy_vs_strength=False, tsne=False, image=False, train_images=False, test_images=False, test_image_iteration=[0], saliency=False, saliency_sample=0, smoothgrad=False, saliency_iteration=[0], image_slider=None):
        p = plot()
        model_name = self.model.get_name()

        if accuracy_vs_epoch:
            #if len(self.accuracy) > 1:
                #print("There are multiple arrays of accuracy. Only showing the first one. Use accuracy_vs_strength to show all")
            p.accuracy_vs_epoch(self.accuracy, self.loss, self.index, model_name)

        if accuracy_vs_strength:
            p.accuracy_vs_strength(self.accuracy, model_name=model_name)


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
