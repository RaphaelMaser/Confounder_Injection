#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import pandas as pd
'''
127.0.0.1:8000:/?token=417c65a720ebe817507356246b10f9a925d3b89cbb60ed50
'''
import copy
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas
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
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import wandb
import datetime
import ray
import os
import pickle
from termcolor import colored

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

    def accuracy_vs_strength(self, results, n_classes=2, ideal=False):
        #print(results)
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        fig.suptitle("Accuracy vs Strength", fontsize=self.fontsize)
        model_name = results["model_name"][0]

        classification_accuracy = results.pivot("epoch", "confounder_strength", "classification_accuracy")
        confounder_accuracy = results.pivot("epoch", "confounder_strength", "confounder_accuracy")

        class_max_acc = classification_accuracy.max(axis='rows')
        class_mean_acc = classification_accuracy.mean(axis='rows')
        if ideal:
            #ideal_df = pd.DataFrame({"ideal":[1/n_classes,1]}, index=[0,1])
            ideal_df = pd.DataFrame({"theoretical upper bound":[1/n_classes,1]}, index=[0,1])
            sbs.lineplot(data=ideal_df, ax=ax, dashes=[(2,2)], palette=["green"])
        class_max_and_mean_acc = pd.concat({"max": class_max_acc, "mean": class_mean_acc}, axis='columns')

        sbs.lineplot(data=class_max_and_mean_acc, markers=True, ax=ax).set(title=f"{model_name}\nClassification Accuracy", ylim=((1/n_classes)-0.05,1.05))#, xlabel="confounder_labels strength in test set" )
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

    def filter_df(self, df, filter):
        for f in filter:
            df = df[df[f]]

    @staticmethod
    def filter(df, reverse=False, filter_unrealistic=False):
        if filter_unrealistic:
            df = df[(df["model"]!="BrNet_DANN_corr")&(df["model"]!="BrNet_DANN_corr_conditioned_0")&(df["model"]!="BrNet_CF_free_labels_corr")&(df["model"]!="BrNet_CF_free_labels_corr_conditioned_0")]

        if not reverse:
            df = df[(df["model"]=="BrNet_DANN_entropy")|(df["model"]=="BrNet")|(df["model"]=="BrNet_DANN_corr")|(df["model"]=="BrNet_DANN_corr_conditioned_0")|(df["model"]=="BrNet_CF_free_features_corr_conditioned_0")|(df["model"]=="BrNet_CF_free_labels_entropy_conditioned_0")|(df["model"]=="BrNet_CF_free_labels_entropy")|(df["model"]=="BrNet_DANN_entropy_conditioned_0")|(df["model"]=="BrNet_CF_free_features_corr")|(df["model"]=="BrNet_CF_free_labels_corr")|(df["model"]=="BrNet_CF_free_labels_corr_conditioned_0")]
        else:
            df = df[(df["model"]!="BrNet_DANN_entropy")&(df["model"]!="BrNet")&(df["model"]!="BrNet_DANN_corr")&(df["model"]!="BrNet_DANN_corr_conditioned_0")&(df["model"]!="BrNet_CF_free_features_corr_conditioned_0")&(df["model"]!="BrNet_CF_free_labels_entropy_conditioned_0")&(df["model"]!="BrNet_CF_free_labels_entropy")&(df["model"]!="BrNet_DANN_entropy_conditioned_0")&(df["model"]!="BrNet_CF_free_features_corr")&(df["model"]!="BrNet_CF_free_labels_corr")&(df["model"]!="BrNet_CF_free_labels_corr_conditioned_0")]
        return df

    @staticmethod
    def plot_heatmap(table, de_correlate_confounder_target, target_domain_samples, ax=None):
        if de_correlate_confounder_target:
            index = 1
        else:
            index = 0
        df = table
        df = df[df["config.de_correlate_confounder_target"] == de_correlate_confounder_target]
        df = df[df["config.target_domain_samples"] == target_domain_samples]
        df = df.pivot_table(index="model", columns="summary_metrics.confounder_strength", values="classification_accuracy")
        df = df.reindex(df[index].sort_values(ascending=False).index)
        if ax==None:
            sbs.heatmap(data=df, annot=True, vmin=0.5, vmax=1)
        else:
            sbs.heatmap(data=df, annot=True, vmin=0.5, vmax=1, ax=ax)

    @staticmethod
    def split_and_plot_heatmaps(df, de_correlate_confounder_target, target_domain_samples):
        fig,ax = plt.subplots(1,2, figsize=(20,4), constrained_layout=True)
        df1 = plot.filter(df, filter_unrealistic=True)
        plot.plot_heatmap(df1, de_correlate_confounder_target, target_domain_samples, ax[0])

        df2 = plot.filter(df, reverse=True)
        plot.plot_heatmap(df2, de_correlate_confounder_target, target_domain_samples, ax[1])

    @staticmethod
    def plot_heatmap_with_mean(df, ax=None):
        df_mean = df.groupby("model")["classification_accuracy"].mean()
        df_mean = pd.DataFrame(df_mean).rename(columns={"classification_accuracy": "mean"})
        df = df.pivot_table(index="model", columns="experiment", values="classification_accuracy")
        df = pd.concat([df, df_mean], axis=1)
        df = df.reindex(df.sort_values(by="mean", ascending=False).index)
        if ax==None:
            sbs.heatmap(data=df, annot=True, vmin=0.5, vmax=None)
        else:
            sbs.heatmap(data=df, annot=True, vmin=0.5, vmax=None, ax=ax)

    @staticmethod
    def split_and_plot_heatmaps_with_mean(df):
        fig,ax = plt.subplots(2,1, figsize=(14,8), constrained_layout=True)
        df1 = plot.filter(df, filter_unrealistic=True)
        plot.plot_heatmap_with_mean(df1, ax[0])

        df2 = plot.filter(df, reverse=True)
        plot.plot_heatmap_with_mean(df2, ax[1])




class wandb_sync:
    def __init__(self):
        pass

    @staticmethod
    def sync_wandb_data(project=None, filters=None, force_reload=False):
        assert(project is not None)
        path = wandb_sync.get_path_from_filters(project, filters)
        t = time.time()
        if (not os.path.isdir(path)) or force_reload:
            os.makedirs(path, exist_ok=True)

            entity = "confounder_in_ml"
            if project==None:
                project = "BrNet on br_net data"

            api = wandb.Api()
            runs = api.runs(entity + "/" + project, filters=filters)

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
            #runs_df.to_csv(f"{project}.csv")
            runs_df.to_pickle(f"{path}/sync.pkl")
            runs_df.to_json(f"{path}/sync.json")
        print(f"Syncing took {time.time() - t} seconds")
        return runs_df

    @staticmethod
    def convert_and_filter_df(project_csv_path, config_filter):
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
                config_dict[c] = {k:project_df["config"][i][c] for k in range(0,len(project_df["history"].iloc[i]["classification_accuracy"]))}


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
    # returns best network from a wandb project and one specific run (marked by batch_date)

    # get runs matching a specific project, batch_date and tag
    @staticmethod
    def get_runs(project=None, filters=None):
        assert(project!=None and filters!=None)

        api = wandb.Api()
        runs = (api.runs(path=f"confounder_in_ml/{project}", filters=filters, order="-summary_metrics.classification_accuracy"))
        return runs

    # returns a list of the best run for every model
    @staticmethod
    def get_best_runs(project=None, filters=None, force_reload=False):
        print("Searching for best runs ...",end=" ")
        t = time.time()
        path = wandb_sync.get_path_from_filters(project, filters)
        file_path = path + "/best_result.pkl"

        # If path does not exist or force_reload is true then the runs are loaded with the wandb api
        if not os.path.exists(file_path) or force_reload:
            runs = wandb_sync.get_runs(project, filters)
            #print(len(runs))
            run_names = []

            for r in runs:
                run_names.append(r.name)
                assert(r.name == r.config["model_name"])
            run_names = list(dict.fromkeys(run_names))
            #print(run_names)

            best_runs = []
            for m in run_names:
                filters_mod = {"config.model_name": m}
                runs = wandb_sync.get_runs(project, filters|filters_mod)
                if len(runs) == 0:
                    raise Exception("get_best_runs: no runs found")
                best_runs.append(runs[0])

            assert(len(best_runs) == len(list(dict.fromkeys(best_runs))))

            print(f"{len(best_runs)} models found in database ...", end=" ")

            best_runs_dir = {"name": [], "run_path":[]}
            for run in best_runs:
                for key in run.config:
                    best_runs_dir[key] = []
                for key in run.summary_metrics:
                    best_runs_dir[key] = []

            keys = best_runs_dir.keys()
            for run in best_runs:
                for key in keys:
                    if key in run.config.keys():
                        best_runs_dir[key].append(run.config[key])
                    elif key in run.summary_metrics.keys():
                        best_runs_dir[key].append(run.summary_metrics[key])
                    elif key == "name":
                        best_runs_dir["name"].append(run.name)
                    elif key == "run_path":
                        best_runs_dir["run_path"].append(run.path)
                    else:
                        best_runs_dir[key].append(None)
            df = pd.DataFrame.from_dict(data=best_runs_dir, orient="columns")

            os.makedirs(path, exist_ok=True)
            if os.path.exists(file_path):
                os.remove(file_path)
            df.to_pickle(file_path)

        # read offline data and return
        try:
            df = pd.read_pickle(path + "/best_result.pkl")
        except:
            raise Exception("get_best_runs: Error in reading file")
        print(f"done ({round(time.time()-t, 3)}s)")
        return df


    @staticmethod
    def create_models_from_runs(project, filters, force_reload=False, load_complete_model=True):
        print("Re-creating models ...", end=" ")
        t = time.time()
        path = wandb_sync.get_path_from_filters(project, filters)

        try:
            #print(path + "/best_result.pkl")
            runs = pd.read_pickle(path + "/best_result.pkl")
        except:
            raise Exception("create_models_from_runs: no best_results.pkl available")

        model_list = []

        for model in runs["model_name"]:
            config = runs.loc[runs["model_name"] == model]
            model_class = config.loc[:,"model_class"].values[0]
            random_int = config.loc[:,"random"].values[0]
            file_name_weights = str(random_int) + ".pt"
            dir_path_weights = os.path.join(path, "weights")
            file_path_weights = os.path.join(dir_path_weights, file_name_weights)

            file_name_model = str(random_int) + ".pt"
            dir_path_model = os.path.join(path, "model")
            file_path_model = os.path.join(dir_path_model, file_name_model)

            if load_complete_model and os.path.exists(file_path_model) and not force_reload:
                model = torch.load(file_path_model)
            else:
                # conditioning is needed for init of model
                conditioning = config.loc[:,"conditioning"].values[0]
                if conditioning == "" or pd.isna(conditioning):
                    conditioning = None

                model_fact = getattr(Models, model_class)
                if issubclass(model_fact, Models.BrNet):
                    model = model_fact()
                elif issubclass(model_fact, Models.BrNet_adversarial):
                    model = model_fact(config.loc[:,"alpha"].values[0], conditioning=conditioning)
                elif issubclass(model_fact, Models.BrNet_adversarial_double):
                    model = model_fact(config.loc[:,"alpha"].values[0], config.loc[:,"alpha2"].values[0], conditioning=conditioning)
                else:
                    raise AssertionError("Did not find any matching model")


                if not os.path.exists(file_path_weights) or force_reload:
                    run_path = config.loc[:,"run_path"].values[0]
                    os.makedirs(dir_path_weights, exist_ok=True)
                    wandb.restore(file_name_weights, run_path=os.path.join(*run_path), root=dir_path_weights)
                    #print(f"Restored file {file_name_weights}")
                model_weights = torch.load(file_path_weights)
                model.load_state_dict(model_weights)
                os.makedirs(dir_path_model, exist_ok=True)
                torch.save(model, file_path_model)
            model.random = random_int
            model_list.append(model)
        print(f"done ({round(time.time()-t, 3)}s)")
        return model_list

    @staticmethod
    def get_path_from_filters(project, filters):
        path = os.getcwd()
        folder = f"wandb/{project}/"
        for f in sorted(filters):
            folder = folder + f"{f}={filters[f]},"
        return os.path.join(path, folder)




class plot_from_wandb:
    def __init__(self, fontsize=18):
        self.fontsize = fontsize
        pass

    def get_from_table(self, file, string):
        df = pd.read_json(file)
        df = {d[string] for d in df["config"]}
        df = [d for d in df]
        df.sort(reverse=True)
        return df

    # if None then newest is taken
    # basically deprecated (only used when experiments are done seperately from the hyperparameters)
    def accuracy_vs_epoch(self, project_csv_path, config_filter, groupby):
        df_list = wandb_sync.convert_and_filter_df(project_csv_path, config_filter)
        title = ""
        for cf in config_filter:
            title = f"{title} \n {cf}={config_filter[cf]}"
        fig, ax = plt.subplots(2,1,figsize=(20,14))
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
        sbs.lineplot(data=classification_accuracy, ax=ax[0]).set(title=f"Classification accuracy\n{title}", ylim=(0.45,1.05))

        if "confounder_accuracy" in df:
            confounder_accuracy = df.pivot("epoch", groupby, "confounder_accuracy")
            sbs.lineplot(data=confounder_accuracy, ax=ax[1]).set(title=f"Confounder accuracy\n{title}", ylim=(0.45,1.05))

        plt.tight_layout()
        return


class CfDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, domain_labels=None, confounder_labels=None, confounder_features=None):
        self.x = x
        self.y = y
        assert(domain_labels != None)
        assert(confounder_labels != None)
        assert(confounder_features != None)

        if domain_labels == None:
            domain_labels = np.fill((len(y)),-1)
        if confounder_labels == None:
            confounder_labels = np.fill((len(y)),-1)

        self.domain_labels = domain_labels
        self.confounder_labels = confounder_labels
        self.confounder_features = confounder_features

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], {'y': self.y[idx], 'domain_labels': self.domain_labels[idx], 'confounder_labels':self.confounder_labels[idx], 'confounder_features': self.confounder_features[idx]}

    def confounder_size(self):
        confounder_samples = self.confounder_labels[self.confounder_labels != -1]
        return len(confounder_samples)


class create_dataloader:
    def __init__(self, x=None, y=None, domain_labels=None, confounder_labels=None, confounder_features = None, batch_size=1):
        #assert(x != None and y != None)
        self.x = x
        self.y = y
        self.domain_labels = domain_labels
        self.confounder_labels = confounder_labels
        self.batch_size = batch_size
        self.confounder_features = confounder_features
        assert(len(x)==len(y)==len(domain_labels)==len(confounder_labels)==len(confounder_features))


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
        tensor_confounder_features= torch.Tensor(self.confounder_features).long()

        dataset = CfDataset(x=tensor_x, y=tensor_y, domain_labels=tensor_domain, confounder_labels=tensor_confounder_labels, confounder_features=tensor_confounder_features)
        #train_dataset, test_dataset = self.split_dataset(dataset)
        # TODO delete unnecessary stuff
        # create DataLoader
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle=True)

        return train_dataloader


class generator:
    def __init__(self, mode, samples, confounding_factor, overlap=0, seed=42, params=None, de_correlate_confounder = False, domain=0):
        np.random.seed(seed)
        self.x = None
        self.y = None
        self.confounder_labels = None
        self.confounder_features = None
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
        if N <= 0:
            return None, None

        if params is None:
            params = [
                [[1, 4], [2, 6]], # real feature
                [[5, 4], [10, 6]] # confounder_labels
            ]

        n_groups = len(params[0])
        self.domain_labels = np.full((N*n_groups), self.domain)
        self.confounder_labels = np.full((N*n_groups), n_groups)
        self.class_labels = np.full((N*n_groups), -1)


        overlap = self.overlap
        assert(overlap % 2 == 0 and overlap <= 32)
        overhang = int(overlap / 2)

        confounded_samples = int(N*self.confounded_samples)

        for g in range(n_groups):
            # define confounder_labels labels
            if self.de_correlate_confounder == True:
                # confounding effects between groups, confounder_labels chosen by chance
                self.confounder_labels[N*g:N*g + confounded_samples] = np.random.randint(0,n_groups, size=confounded_samples)
            else:
                # confounding effects between groups
                self.confounder_labels[N*g:N*g + confounded_samples] = np.full((confounded_samples),g)

            # define class labels
            self.class_labels[N*g:N*(g+1)] = np.full((N),g)

        # cf and mc are the confounding- and class-features
        # these are derived from the confounding -and class-labels
        cf = np.zeros((N*n_groups))
        mf = np.zeros((N*n_groups))

        # derive cf from confounder_labels-label
        for i in range(0,len(self.confounder_labels)):
            index = self.confounder_labels[i]
            # check that index is valid (else the value stays at zero)
            if index != n_groups:
                #print("Index is ",index)
                cf[i] = np.random.uniform(params[1][index][0], params[1][index][1])
            #else:
                #cf[i] = np.nan TODO

        # derive mf from class-label
        for g in range(n_groups):
            mf[N*g:N*(g+1)] = np.random.uniform(params[0][g][0], params[0][g][1], N)

        # simulate images
        x = np.zeros((N*n_groups,1,32,32))
        for i in range(0,N*n_groups):
            # add real feature
            x[i,0,:16 + overhang, :16 + overhang] += self.gkern(kernlen=16+overhang, nsig=5) * mf[i]
            x[i,0, 16 - overhang:, 16 - overhang:] += self.gkern(kernlen=16+overhang, nsig=5) * mf[i]

            # check if confounder_labels feature should be added
            #if not np.isnan(cf[i]): TODO
            x[i,0, 16 - overhang:, :16 + overhang] += self.gkern(kernlen=16+overhang, nsig=5) * cf[i]
            x[i,0,:16 + overhang,16 - overhang:] += self.gkern(kernlen=16+overhang, nsig=5) * cf[i]

            # add random noise
            x[i] = x[i] + np.random.normal(0,0.01,size=(1,32,32))

        # if there is no confounder in the image then we need to create some arbitrary values
        # otherwise our model gets a tensor with nan as input
        if np.isnan(cf).any():
            raise Exception("NaN in cf feature vector")
            #self.confounder_features = np.full((N*n_groups), 0)
        #else:
        self.confounder_features = cf

        # if self.conditioning != -1:
        #     self.domain_labels[N*self.conditioning:N*(self.conditioning+1)] = -1
        #     self.confounder_labels[N*self.conditioning:N*(self.conditioning+1)] = -1
        #     self.confounder_features[N*self.conditioning:N*(self.conditioning+1)] = -1

        if self.debug:
            print("--- generator ---")
            print("Confounding factor:",self.confounded_samples)
            print("Number of samples per group")
            print("Confounded samples per group (estimate):", confounded_samples)

        self.x = x
        self.y = self.class_labels


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
        return self.x, self.y, self.domain_labels, self.confounder_labels, self.confounder_features


class confounder_injection:

    def __init__(self, path):
        pass

    def inject_local_xy(self):
        pass

    def read_dataset(self, path):
        pass

class train:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.accuracy = []
        self.loss = []

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

                class_pred, _, _ = self.model(X)
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

    def train_normal(self, train_dataloader, optimizer):
        self.model = self.model.to(self.device)

        self.model.train()
        for batch, (X,label) in enumerate(train_dataloader):
            X = X.to(self.device)
            y = label["y"].to(self.device)

            # Compute prediction error
            class_pred, _, _ = self.model(X)

            # check if tensors are valid
            self.check_for_nan([class_pred, y])

            # compute loss
            loss = self.model.loss(class_pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return

    def train_adversarial(self, train_dataloader, optimizer):
        adversarial_mode = self.model.mode
        self.model = self.model.to(self.device)

        # loss functions
        class_loss_function = self.model.loss
        adversarial_loss_function = self.model.adv_loss

        self.model.train()
        for batch, (X,label) in enumerate(train_dataloader):
            X = X.to(self.device)
            y = label["y"].to(self.device)
            adversary_label = label[adversarial_mode].to(self.device)

            # print(f"--- train_adversarial ---\n"
            #       f"y:{y}\n"
            #       f"adversary_label:{adversary_label}")

            # prediction
            class_pred, adversary_pred, _ = self.model(X)
            #print(f"class_pred: {class_pred}")
            #print(f"class_real: {y}")
            #print(f"adversary_pred: {adversary_pred}")
            adversary_label, adversary_pred = self.condition_and_filter(y=y, real=adversary_label, pred=adversary_pred, condition=self.model.conditioning)
            #print("--- filtering ---")
            # print(f"adversary_label: {adversary_label}")
            # print(f"adversary_pred: {adversary_pred}\n\n")
            #raise Exception

            # check if tensors are valid
            self.check_for_nan([adversary_label, adversary_pred, class_pred, y])

            # compute error
            class_loss = class_loss_function(class_pred, y)
            adversary_loss = adversarial_loss_function(adversary_pred, adversary_label)
            loss = class_loss + adversary_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        return

    def train_adversarial_double(self, train_dataloader, optimizer):
        adversary_mode = self.model.mode
        adversary2_mode = self.model.mode2
        self.model = self.model.to(self.device)

        # loss functions
        class_loss_function = self.model.loss
        adversary_loss_function = self.model.adv_loss
        adversary2_loss_function = self.model.adv2_loss

        self.model.train()
        for batch, (X,label) in enumerate(train_dataloader):
            X = X.to(self.device)
            y = label["y"].to(self.device)
            adversary_label = label[adversary_mode].to(self.device)
            adversary2_label = label[adversary2_mode].to(self.device)

            # prediction
            class_pred, adversary_pred, adversary2_pred = self.model(X)
            adversary_label, adversary_pred, adversary2_label, adversary2_pred = self.condition_and_filter_double(y=y, real=adversary_label, pred=adversary_pred, real2=adversary2_label, pred2=adversary2_pred, condition=self.model.conditioning)

            # check if tensors are valid
            self.check_for_nan([adversary_label, adversary_pred, adversary2_pred, adversary2_label, class_pred, y])

            # compute error
            class_loss = class_loss_function(class_pred, y)
            adversary_loss = adversary_loss_function(adversary_pred, adversary_label)
            adversary2_loss = adversary2_loss_function(adversary2_pred, adversary2_label)
            loss = class_loss + adversary_loss + adversary2_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return

    def check_for_nan(self, array):
        for tensor in array:
            if torch.isnan(tensor).any():
                raise Exception(f"nan in input data detected\narray with nan:{array}\n")

    def run(self, train_dataloader, optimizer):
        if self.model.adversarial:
            if isinstance(self.model, Models.BrNet_adversarial_double):
                self.train_adversarial_double(train_dataloader=train_dataloader, optimizer=optimizer)
            else:
                self.train_adversarial(train_dataloader=train_dataloader, optimizer=optimizer)
        else:
            self.train_normal(train_dataloader=train_dataloader, optimizer=optimizer)

        return

    def condition_and_filter(self, y, real, pred, condition):
        #assert(real != -1)
        if condition != None:
            filtered_real = real[y == condition]
            filtered_pred = pred[y == condition]
        else:
            filtered_real = real
            filtered_pred = pred

        return filtered_real, filtered_pred

    def condition_and_filter_double(self, y, real, pred, real2, pred2, condition):
        #assert(real != -1)
        if condition != None:
            filtered_real = real[y == condition]
            filtered_pred = pred[y == condition]
            filtered_real2 = real2[y == condition]
            filtered_pred2 = pred2[y == condition]
        else:
            filtered_real = real
            filtered_pred = pred
            filtered_real2 = real2
            filtered_pred2 = pred2

        return filtered_real, filtered_pred, filtered_real2, filtered_pred2


class confounder:
    all_results = pd.DataFrame()
    t = None
    def __init__(self, seed=41, mode="NeuralNetwork", debug=False, clean_results=False, start_timer=False, tune=False, name=None):
        torch.backends.cudnn.benchmark = True
        self.mode = mode
        self.seed = seed
        self.random_numbers = np.random.default_rng()

        self.train_x = None
        self.train_y = None
        self.train_domain_labels = None
        self.train_confounder_labels = None
        self.train_confounder_features = None

        self.test_x = None
        self.test_y = None
        self.test_domain_labels = None
        self.test_confounder_labels = None
        self.test_confounder_features = None

        self.results = None
        self.loss = []
        self.debug = debug
        self.index = []
        self.fontsize = 18
        self.model_title_add = ""
        self.tune = tune
        self.name = name
        self.wandb_sweep_init = None


        if start_timer:
            confounder.t = time.time()

        if clean_results:
            confounder.all_results= pd.DataFrame()

        if debug:
            print("--- constructor ---")
            #print("Model:\n",model)

    def reset_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def generate_data(self, mode=None, overlap=0, samples=512, test_samples=512, target_domain_samples=0, target_domain_confounding=0, train_confounding=1, test_confounding=[1], de_correlate_confounder_test=False, de_correlate_confounder_target=False, params=None):
        self.reset_seed()

        iterations = len(test_confounding)
        self.n_classes = len(params[0])
        self.train_x = np.empty((iterations,samples*self.n_classes + target_domain_samples*self.n_classes,1,32,32))
        self.test_x = np.empty((iterations,test_samples*self.n_classes,1,32,32))

        self.train_y = np.empty((iterations,samples*self.n_classes + target_domain_samples*self.n_classes))
        self.test_y = np.empty((iterations,test_samples*self.n_classes))

        self.train_domain_labels = np.empty((iterations,samples*self.n_classes + target_domain_samples*self.n_classes))
        self.test_domain_labels = np.empty((iterations,test_samples*self.n_classes))

        self.train_confounder_labels = np.empty((iterations,samples*self.n_classes + target_domain_samples*self.n_classes))
        self.test_confounder_labels = np.empty((iterations,test_samples*self.n_classes))

        self.train_confounder_features = np.empty((iterations,samples*self.n_classes + target_domain_samples*self.n_classes))
        self.test_confounder_features = np.empty((iterations,test_samples*self.n_classes))

        self.samples = samples
        self.target_domain_samples = target_domain_samples
        self.overlap = overlap
        self.train_confounding = train_confounding
        self.test_confounding = test_confounding
        self.target_domain_confounding = target_domain_confounding
        self.de_correlate_confounder_test = de_correlate_confounder_test
        self.de_correlate_confounder_target =de_correlate_confounder_target
        self.params = params



        self.index = test_confounding

        # train data
        g_train = generator(mode=mode, samples=samples, overlap=overlap, confounding_factor=train_confounding, params=params, domain=0)
        g_train_data = g_train.get_data()
        self.train_x[0,:samples*self.n_classes] = g_train_data[0]
        self.train_y[0,:samples*self.n_classes] = g_train_data[1]
        self.train_domain_labels[0,:samples*self.n_classes] = g_train_data[2]
        self.train_confounder_labels[0,:samples*self.n_classes] = g_train_data[3]
        self.train_confounder_features[0,:samples*self.n_classes] = g_train_data[4]

        # append target domain_labels data to source domain_labels data
        if target_domain_samples != 0:
            g_target_domain = generator(mode=mode, samples=target_domain_samples, overlap=overlap, confounding_factor=target_domain_confounding, params=params, domain=1, de_correlate_confounder=de_correlate_confounder_target)
            g_target_domain_data =g_target_domain.get_data()
            self.train_x[0,samples*self.n_classes:] = g_target_domain_data[0]
            self.train_y[0,samples*self.n_classes:] = g_target_domain_data[1]
            self.train_domain_labels[0,samples*self.n_classes:] = g_target_domain_data[2]
            self.train_confounder_labels[0,samples*self.n_classes:] = g_target_domain_data[3]
            self.train_confounder_features[0,samples*self.n_classes:] = g_target_domain_data[4]

        i = 0
        for cf_var in test_confounding:
            # test data
            g_test = generator(mode=mode, samples=test_samples, overlap=overlap, confounding_factor=cf_var, params=params, domain=0, de_correlate_confounder=de_correlate_confounder_test)
            g_test_data =g_test.get_data()
            self.test_x[i] = g_test_data[0]
            self.test_y[i] = g_test_data[1]
            self.test_domain_labels[i] = g_test_data[2]
            self.test_confounder_labels[i] = g_test_data[3]
            self.test_confounder_features[i] = g_test_data[4]


            i += 1

            if self.debug:
                print("--- generate_data ---")
                #print("Generated Data of dimension ", self.train_x.shape)
            # print(f"--- Data generation ---\n"
            #       f"y:{self.train_y}\n"
            #       f"confounder_labels:{self.train_confounder_labels}\n"
            #       f"confounder_features:{self.train_confounder_features}\n"
            #       f"domain_labels:{self.train_domain_labels}")
        return self.train_x, self.train_y, self.test_x, self.test_y

    def inject_confounder(self, type=None):
        pass

    def train(self, use_tune=False, use_wandb=False, wandb_sweep=False, model=Models.NeuralNetwork(32 * 32), epochs=1, device ="cuda", optimizer = None, hyper_params=None, wandb_init=None):
        self.reset_seed()
        name = model.get_name()
        self.model = copy.deepcopy(model)
        mode = "offline"
        working_directory = os.getcwd()
        start_epoch = 1
        os.environ['WANDB_MODE'] = mode
        os.environ['WANDB_SILENT'] = "true"

        # for f in os.listdir(working_directory):
        #     os.remove(os.path.join(working_directory,f))
        # if os.path.exists(os.path.join(working_directory,"wandb")):
        #     shutil.rmtree(os.path.join(working_directory,"wandb"))
        checkpoint = session.get_checkpoint()
        if checkpoint and wandb_init.get("pbt"):
            #warnings.warn(colored("CHECKPOINT FOUND","red"))
            with checkpoint.as_directory() as checkpoint_dir:
                #state = torch.load(os.path.join(checkpoint_dir,"checkpoint.pt"))
                state = torch.load(checkpoint_dir)
                self.model.load_state_dict(state["model_state_dict"])
                start_epoch = state["step"]
                #warnings.warn(f"NEW EPOCH:{start_epoch}")

        if device == "cuda":
            if torch.cuda.is_available():
                print("CUDA detected")
            else:
                device="cpu"

        # if wandb_init["dir"] == None:
        #     wandb_init["dir"] = os.getcwd()
        #
        # batch_size_temp =hyper_params["batch_size"]
        # print(f"\n--- Debugging ---\n"
        #       f"Model={self.model.get_name()}\n"
        #       f"Train_length={len(self.train_y[0])}\n"
        #       f"Batch_size={batch_size_temp}")

        if len(self.train_y[0]) < hyper_params["batch_size"]:
            # print(f"Batch_size was too high with length of array={len(self.train_y[0])} and batch_size={batch_size_temp}\n")
            hyper_params["batch_size"] = len(self.train_y[0])
            if wandb_init.get("finetuning") != 1:
                raise Exception("Batch_size had to be reset although finetuning is off. Choose a correct batch_size")


        config = {
            "model_name":name,
            "model_class":type(self.model).__name__,
            "epochs":epochs,
            "device": device,
            "optimizer": optimizer,
            "loss": model.loss,
            "adversarial_loss": model.adv_loss,
            "samples": self.samples,
            "target_domain_samples": self.target_domain_samples,
            "overlap": self.overlap,
            "train_confounding": self.train_confounding,
            "test_confounding": self.test_confounding,
            "target_domain_confounding": self.target_domain_confounding,
            "de_correlate_confounder_test": self.de_correlate_confounder_test,
            "de_correlate_confounder_target": self.de_correlate_confounder_target,
            "params": self.params,
            "seed": self.seed,
            "random": self.random_numbers.integers(sys.maxsize)
        }


        if use_wandb:
            if "project" not in wandb_init:
                wandb_init["project"] = "None"
            if "group" not in wandb_init:
                wandb_init["group"] = "None"
            if "date" not in wandb_init:
                wandb_init["date"] = "None"
            if "batch_date" not in wandb_init:
                wandb_init["batch_date"] = "None"
            config["finetuning"] = wandb_init.get("finetuning")
            config["pbt"] = wandb_init.get("pbt")
            config["date"] = wandb_init["date"]
            config["batch_date"] = wandb_init["batch_date"]

        # if we use wandb sweep the hyperparams are already set by wandb
        if not wandb_sweep:
            config["alpha"] = self.model.alpha
            config["alpha2"] = self.model.alpha2
            config["lr"] = hyper_params["lr"]
            config["weight_decay"] = hyper_params["weight_decay"]
            config["batch_size"] = hyper_params["batch_size"]

        if hasattr(model, "conditioning"):
            config["conditioning"] = self.model.conditioning
        if hasattr(model, "mode"):
            config["adversary_mode"] = self.model.mode
        if hasattr(model, "mode2"):
            config["adversary2_mode"] = self.model.mode2

        if use_wandb:
            # id=os.path.basename(working_directory),
            wandb.init(name=name, resume=True, entity="confounder_in_ml", config=config, project=wandb_init["project"], group=wandb_init["group"], reinit=False, settings=wandb.Settings(start_method="fork"), mode=mode, dir=working_directory)
            time.sleep(5)
            config = wandb.config

        if wandb_sweep:
            # setting hyperparameters
            if "alpha" in config:
                self.model.alpha = wandb.config["alpha"]
            if "alpha2" in config:
                self.model.alpha2 = wandb.config["alpha2"]

        delta_t = time.time()
        set = 0
        results = {"confounder_strength":[],"model_name":[],"epoch":[],"classification_accuracy":[], "confounder_accuracy":[]}

        if hyper_params == None:
            if not wandb_sweep:
                raise AssertionError("Choose some hyperparameter for the optimizer")
        # else:
        #     if not 'weight_decay' in config:
        #         config['weight_decay'] = 0

        model_optimizer = optimizer(params=self.model.parameters(), lr=config['lr'], weight_decay=config["weight_decay"])
        train_dataloader = create_dataloader(self.train_x[set], self.train_y[set], domain_labels=self.train_domain_labels[set], confounder_labels=self.train_confounder_labels[set], batch_size=config["batch_size"], confounder_features=self.train_confounder_features[set]).get_dataloader()

        test_dataloader = []
        for cf_var in range(0,len(self.index)):
            test_dataloader.append(create_dataloader(self.test_x[cf_var], self.test_y[cf_var], domain_labels=self.test_domain_labels[cf_var], confounder_labels=self.test_confounder_labels[cf_var], batch_size=config["batch_size"], confounder_features=self.test_confounder_features[cf_var]).get_dataloader())

        for epoch in range(start_epoch, epochs+1):
            # load new results

            t = train(self.model, device)
            t.run(train_dataloader, model_optimizer)

            for cf_var in range(0,len(self.index)):
                classification_accuracy, confounder_accuracy = t.test(test_dataloader[cf_var])

                results["confounder_strength"].append(self.index[cf_var])
                results["model_name"].append(self.model.get_name()+self.model_title_add)
                results["epoch"].append(epoch)
                results["classification_accuracy"].append(classification_accuracy)
                results["confounder_accuracy"].append(confounder_accuracy)

                if use_wandb:
                    wandb.log({"classification_accuracy":classification_accuracy, "confounder_accuracy":confounder_accuracy, "confounder_strength":self.index[cf_var], "epoch":epoch}, step=epoch)

                # elif use_wandb:
                #     wandb.log({"classification_accuracy":classification_accuracy, "confounder_accuracy":confounder_accuracy, "confounder_strength":self.index[cf_var], "epoch":epoch}, commit=False, step=epoch)

                # register accuracy in use_tune
                if use_tune:
                    checkpoint = None

                    if epoch % 10 == 0 and wandb_init.get("pbt"):
                        # PBT needs checkpointing
                        # create checkpoint file
                        #path = os.path.join(working_directory,"model")
                        path = "model/checkpoint.pt"
                        #os.makedirs(path, exist_ok=True)
                    # save state to checkpoint file
                        torch.save(
                            {
                                "step": epoch,
                                "model_state_dict":model.state_dict(),
                                "mean_accuracy":classification_accuracy
                            },
                            #os.path.join(path,"checkpoint.pt"),
                            path
                        )
                        checkpoint = Checkpoint.from_directory(path)

                    # report to tune
                    session.report(metrics={"mean_accuracy":classification_accuracy, "epoch":epoch}, checkpoint=checkpoint)

        if use_wandb:
            # save model parameters and upload to wandb
            path = os.path.join(working_directory, str(config["random"]) + ".pt")
            torch.save(self.model.state_dict(), path)
            wandb.save(path)

            #wandb.log()
            wandb.config.update({"trained_model": self.model}, allow_val_change=True)
            wandb.finish()

            if mode == "offline":# and wandb_init["dir"] == None:
                t = time.time()
                print(f"--- syncing ---\n"
                      f"current_dir={working_directory}\n"
                      f"files={working_directory}")
                os.system(f"conda run -n confounder_3.10 wandb sync --sync-all")
                #wandb_path = os.path.join("wandb",f"{wandb.run.id}")
                #os.system(f"conda run -n confounder_3.10 wandb sync --id {wandb_path}")
                print(f"--- took {time.time()-t}s")


        self.results = pd.DataFrame(results)
        confounder.all_results = pd.concat([confounder.all_results, self.results], ignore_index=True)
        #confounder_labels.all_results.append(self.results)

        if self.debug:
            print("--- train ---")
            print("Training took ",time.time() - delta_t, "s")

        return self.results

    def test(self, batch_size=1):
        assert(len(self.test_x) == 1)
        # create test dataloader
        test_dataloader = create_dataloader(self.test_x[0], self.test_y[0], domain_labels=self.test_domain_labels[0], confounder_labels=self.test_confounder_labels[0], batch_size=batch_size, confounder_features=self.test_confounder_features[0]).get_dataloader()
        t = train(self.model)
        classification_accuracy, confounder_accuracy = t.test(test_dataloader)
        return classification_accuracy, confounder_accuracy

    def test_best_networks(self, project="Hyperparameters", filters=None, force_reload=False, load_complete_model=True, experiment=0):
        t = time.time()
        # get best runs
        wandb_sync.get_best_runs(project=project, filters=filters, force_reload=force_reload)

        # get models
        model_list = wandb_sync.create_models_from_runs(project=project, filters=filters, force_reload=force_reload, load_complete_model=load_complete_model)

        # test networks and save accuracy
        results = {"model":[], "classification_accuracy":[], "confounder_accuracy":[], "random":[], "experiment":[experiment for x in range(len(model_list))]}
        keys = filters.keys()
        for k in keys:
            results[k] = []

        for m in model_list:
            self.model = m
            acc = self.test(batch_size=64)
            results["model"].append(m.get_name())
            results["classification_accuracy"].append(acc[0])
            results["confounder_accuracy"].append(acc[1])
            for k in keys:
                results[k].append(filters[k])
            results["random"].append(self.model.random)

        print(f"Runs synced, models re-created and tested (took {round(time.time()-t, 3)}s)\n")

        results_df = pandas.DataFrame.from_dict(results)
        return results_df


    def plot(self, accuracy_vs_epoch=False, accuracy_vs_strength=False, tsne=False, image=False, train_images=False, test_images=False, test_image_iteration=[0], saliency=False, saliency_sample=0, smoothgrad=False, saliency_iteration=[0], image_slider=None, plot_all=False, epoch_vs_strength_ideal=False):
        p = plot()
        model_name = self.model.get_name()

        if accuracy_vs_epoch:
            p.accuracy_vs_epoch(self.results)

        if accuracy_vs_strength:
            p.accuracy_vs_strength(self.results, n_classes=self.n_classes, ideal=epoch_vs_strength_ideal)

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
        class_pred, domain_pred, _ = self.model(x)

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


class helper():
    def __init__(self):
        pass


    @staticmethod
    def BrNet_on_BrNet_data(batch_date, test_confounding, target_domain_samples, target_domain_confounding, de_correlate_confounder_target, force_reload=False, seed=None, load_complete_model=True, experiment=0, finetuning=None):
        print(f"Experiment {experiment}:"
              f"\n-- batch_date={batch_date}"
              f"\n-- test_confounding={test_confounding}"
              f"\n-- target_domain_samples={target_domain_samples}"
              f"\n-- target_domain_confounding={target_domain_confounding}"
              f"\n-- de_correlate_confounder_target={de_correlate_confounder_target}")

        params = [
            [[1, 4], [3, 6]], # real feature
            [[10, 12], [20, 22]] # confounder_labels
        ]

        c = confounder(seed=seed)
        c.generate_data(mode="br_net", samples=512, target_domain_samples=target_domain_samples, target_domain_confounding=target_domain_confounding, train_confounding=1, test_confounding=[test_confounding], de_correlate_confounder_target=de_correlate_confounder_target, de_correlate_confounder_test=de_correlate_confounder_target, params=params)

        filters = {
            "summary_metrics.confounder_strength": test_confounding,
            "config.target_domain_samples": target_domain_samples,
            "config.target_domain_confounding": target_domain_confounding,
            "config.de_correlate_confounder_target": de_correlate_confounder_target,
            "config.batch_date": batch_date,
        }

        if finetuning != None:
            print(f"-- finetuning={finetuning}")
            filters["config.finetuning"] = finetuning

        df = c.test_best_networks(filters=filters, force_reload=force_reload, load_complete_model=load_complete_model, experiment=experiment)
        return df

    @staticmethod
    def BrNet_on_BrNet_data_all(batch_date, force_reload=False, seed=918291, load_complete_model=True, legacy=True):
        t = time.time()
        if legacy:
            experiments = [
                helper.BrNet_on_BrNet_data(batch_date=batch_date, test_confounding=0, target_domain_samples=0, target_domain_confounding=0, de_correlate_confounder_target=0, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="1.\nno confounder\n0/512"),#1),
                #helper.BrNet_on_BrNet_data(batch_date=batch_date, test_confounding=1, target_domain_samples=0, target_domain_confounding=0, de_correlate_confounder_target=0, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="no-confounder\n16/512"),#2),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, test_confounding=0, target_domain_samples=16, target_domain_confounding=0, de_correlate_confounder_target=0, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="2.\nno confounder\n16/512"),#2),
                #helper.BrNet_on_BrNet_data(batch_date=batch_date, test_confounding=1, target_domain_samples=16, target_domain_confounding=0, de_correlate_confounder_target=0, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment=4),

                helper.BrNet_on_BrNet_data(batch_date=batch_date, test_confounding=1, target_domain_samples=0, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="3.\nde-correlated\n0/512"),#3),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, test_confounding=1, target_domain_samples=16, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="4.\nde-correlated\n16/512"),#4),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, test_confounding=1, target_domain_samples=64, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="5.\nde-correlated\n64/512"),#5),
            ]
        else:
            experiments = [
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=0, test_confounding=0, target_domain_samples=0, target_domain_confounding=0, de_correlate_confounder_target=0, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="1.\nno-confounder\n0/512"),#1),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=0, test_confounding=0, target_domain_samples=8, target_domain_confounding=0, de_correlate_confounder_target=0, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="2.\nno-confounder\n8/512"),#2),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=0, test_confounding=1, target_domain_samples=0, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="3.\nde-correlated\n0/512"),#4),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=0, test_confounding=1, target_domain_samples=8, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="4.\nde-correlated\n8/512"),#5),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=0, test_confounding=1, target_domain_samples=16, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="5.\nde-correlated\n16/512"),#5),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=0, test_confounding=1, target_domain_samples=64, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="6.\nde-correlated\n64/512"),#7),

                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=1, test_confounding=0, target_domain_samples=8, target_domain_confounding=0, de_correlate_confounder_target=0, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="1.\nno-confounder\n8/512\nfinetuning"),#),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=1, test_confounding=1, target_domain_samples=8, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="2.\nde-correlated\n8/512\nfinetuning"),#6),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=1, test_confounding=1, target_domain_samples=16, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="3.\nde-correlated\n16/512\nfinetuning"),#6),
                helper.BrNet_on_BrNet_data(batch_date=batch_date, finetuning=1, test_confounding=1, target_domain_samples=64, target_domain_confounding=1, de_correlate_confounder_target=1, force_reload=force_reload, seed=seed, load_complete_model=load_complete_model, experiment="4.\nde-correlated\n64/512\nfinetuning"),#8),

            ]

        df = pd.concat(experiments)
        print(f"--- Synced and processed all experiments (took {round(time.time()-t, 3)}s) ---")
        return df.reset_index(drop=True)

