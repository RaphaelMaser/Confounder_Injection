from torch import nn
from Framework.Layers import GradientReversal
from scipy import stats
import torch
import pandas as pd

def reset_seed():
    torch.manual_seed(42)

# Building a Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        reset_seed()
        super(NeuralNetwork, self).__init__()
        self.alpha = None
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            #nn.ReLU(),
            #nn.Linear(64, 2),
            #nn.ReLU(),
            #nn.Linear(16, 2),
            #nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits, None

    def get_name(self):
        return "Neural Network"

# In[11]:


# Building a Neural Network architecture
class SimpleConv(nn.Module):
    def __init__(self):
        reset_seed()
        super(SimpleConv, self).__init__()
        self.alpha = None
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(1176,84),
            nn.ReLU(),

            nn.Linear(84,2)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits, None

    def get_name(self):
        return "SimpleConv"

# Building a Neural Network architecture
class LeNet_5(nn.Module):
    def __init__(self):
        reset_seed()
        super(LeNet_5, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16,kernel_size=5),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(120,84),
            nn.ReLU(),

            nn.Linear(84,2)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits, None

    def get_name(self):
        return "LeNet"

# In[12]:


# Building a Neural Network architecture
class BrNet(nn.Module):
    def __init__(self, n_classes=2):
        reset_seed()
        super(BrNet, self).__init__()
        self.alpha = None
        self.alpha2 = None
        self.adversarial = False
        self.name = "BrNet"
        self.loss = nn.CrossEntropyLoss()
        self.adv_loss = None
        self.conditioning = None
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(2, 4,kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(4, 8, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(32,n_classes),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits, None, None

    def get_name(self):
        return self.name

class BrNet_adversarial(nn.Module):
    def __init__(self, alpha, class_output, adv_output, conditioning=None):
        reset_seed()
        super(BrNet_adversarial, self).__init__()
        self.alpha = alpha
        self.alpha2 = None
        self.adversarial = True
        self.name = "BrNet_adversarial"
        self.mode = None
        self.loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.CrossEntropyLoss()
        self.conditioning = conditioning
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(2, 4,kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(4, 8, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
        )

        self.class_predictor = nn.Sequential(
            nn.Linear(32, class_output)
        )

        self.adv_predictor = nn.Sequential(
            nn.Linear(32, adv_output)
        )

    def forward(self, x):
        features = self.linear_relu_stack(x)
        reverse_features = GradientReversal.apply(features, self.alpha)

        class_features = self.class_predictor(features)
        domain_features = self.adv_predictor(reverse_features)
        return class_features, domain_features, None

    def get_name(self):
        if self.conditioning != None:
            return self.name + f"_conditioned_{int(self.conditioning)}"
        return self.name

class BrNet_CF_free_labels_entropy(BrNet_adversarial):
    def __init__(self, alpha, n_classes=2, conditioning=None):
        reset_seed()
        super().__init__(alpha, n_classes, n_classes+1, conditioning)
        self.name = "BrNet_CF-net_labels_entropy"
        self.mode = "confounder_labels"
        self.adversarial = True

class BrNet_CF_free_labels_corr(BrNet_adversarial):
    def __init__(self, alpha, n_classes=2, conditioning=None):
        reset_seed()
        super().__init__(alpha, n_classes, 1, conditioning)
        self.name = "BrNet_CF-net_labels_corr"
        self.mode = "confounder_labels"
        self.adversarial = True
        self.adv_loss = squared_correlation()
        self.adv_output = 1

class BrNet_CF_free_features_corr(BrNet_adversarial):
    def __init__(self, alpha, n_classes=2, conditioning=None):
        reset_seed()
        super().__init__(alpha, n_classes, 1, conditioning)
        self.name = "BrNet_CF-net_features_corr"
        self.mode = "confounder_features"
        self.adversarial = True
        self.adv_loss = squared_correlation()
        self.adv_output = 1

class BrNet_DANN_entropy(BrNet_adversarial):
    def __init__(self, alpha, n_classes=2, conditioning=None):
        reset_seed()
        super().__init__(alpha, n_classes, 2, conditioning)
        self.name = "BrNet_DANN_entropy"
        self.mode = "domain_labels"
        self.adversarial = True

class BrNet_DANN_corr(BrNet_adversarial):
    def __init__(self, alpha, n_classes=2, conditioning=None):
        reset_seed()
        super().__init__(alpha, n_classes, 1, conditioning)
        self.name = "BrNet_DANN_corr"
        self.mode = "domain_labels"
        self.adversarial = True
        self.adv_loss = squared_correlation()
        self.adv_output = 1

class BrNet_adversarial_double(nn.Module):
    def __init__(self, alpha, alpha2, class_output, adv1_output, adv2_output, conditioning=None):
        reset_seed()
        super(BrNet_adversarial_double, self).__init__()
        self.alpha = alpha
        self.alpha2 = alpha2
        self.adversarial = False
        self.name = "BrNet_adversarial_two"
        self.mode = None
        self.mode2 = None
        self.loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.CrossEntropyLoss()
        self.adv2_loss = nn.CrossEntropyLoss()
        self.conditioning = conditioning
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(2, 4,kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(4, 8, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
        )

        self.class_predictor = nn.Sequential(
            nn.Linear(32, class_output)
        )

        self.adv_predictor = nn.Sequential(
            nn.Linear(32, adv1_output)
        )

        self.adv2_predictor = nn.Sequential(
            nn.Linear(32, adv2_output)
        )

    def forward(self, x):
        features = self.linear_relu_stack(x)
        reverse_features = GradientReversal.apply(features, self.alpha)

        class_features = self.class_predictor(features)
        adv_features = self.adv_predictor(reverse_features)
        adv2_features = self.adv2_predictor(reverse_features)
        return class_features, adv_features, adv2_features

    def get_name(self):
        if self.conditioning != None:
            return self.name + f"_conditioned_{int(self.conditioning)}"
        return self.name

class BrNet_CF_free_DANN_labels_entropy(BrNet_adversarial_double):
    def __init__(self, alpha, alpha2, n_classes=2, conditioning=None):
        reset_seed()
        super().__init__(alpha, alpha2, n_classes, 2, n_classes+1, conditioning)
        self.name = "BrNet_CF-net_DANN_labels_entropy"
        self.mode = "domain_labels"
        self.mode2 = "confounder_labels"
        self.adversarial = True
        self.adv_output = 1

class BrNet_CF_free_DANN_labels_entropy_features_corr(BrNet_adversarial_double):
    reset_seed()
    def __init__(self, alpha, alpha2, n_classes=2, conditioning=None):
        super().__init__(alpha, alpha2, n_classes, 2, 1, conditioning)
        self.name = "BrNet_CF-net_DANN_labels_entropy_features_corr"
        self.mode = "domain_labels"
        self.mode2 = "confounder_features"
        self.adversarial = True
        #self.adv_loss = squared_correlation()
        self.adv2_loss = squared_correlation()
        self.adv_output = 1


class SimpleConv_DANN(nn.Module):
    def __init__(self, alpha):
        super(SimpleConv_DANN, self).__init__()
        reset_seed()
        self.alpha = alpha
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(1176,84),
            nn.ReLU(),

        )

        self.class_predictor = nn.Sequential(
            nn.Linear(84,2)
        )

        self.domain_predictor = nn.Sequential(
            nn.Linear(84,2)
        )

    def forward(self, x):
        features = self.linear_relu_stack(x)
        reverse_features = GradientReversal.apply(features, self.alpha)

        class_features = self.class_predictor(features)
        domain_features = self.domain_predictor(reverse_features)
        return class_features, domain_features

    def get_name(self):
        return "SimpleConv_DANN"

class SimpleConv_CF_free(nn.Module):
    def __init__(self, alpha):
        super(SimpleConv_CF_free, self).__init__()
        reset_seed()
        self.alpha = alpha
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(1176,84),
            nn.ReLU(),

        )

        self.class_predictor = nn.Sequential(
            nn.Linear(84,2)
        )

        self.domain_predictor = nn.Sequential(
            nn.Linear(84,2)
        )

    def forward(self, x):
        features = self.linear_relu_stack(x)
        reverse_features = GradientReversal.apply(features, self.alpha)

        class_features = self.class_predictor(features)
        domain_features = self.domain_predictor(reverse_features)
        return class_features, domain_features

    def get_name(self):
        return "SimpleConv_CF_free"

class squared_correlation(torch.nn.Module):
    def __init__(self):
        super(squared_correlation,self).__init__()

    def forward(self, pred, real):
        #print("--- squared_correlation ---")
        # print(f"Real tensor: {real}")
        # print(f"Pred tensor: {pred}")
        real = real.reshape(len(real),1)
        pred = torch.squeeze(pred)
        real = torch.squeeze(real)
        # print(f"\n\n pred is {pred}\n\n")
        # print(f"\n\n real is {real}\n\n")
        real, pred = self.check_correctness(real=real, pred=pred)
        x = torch.stack((pred, real), dim=0)
        # print(f"\n\n x is {x}\n\n")

        # could happen in conditioning case
        # if len(pred) == 0:
        #     return 0
        corr_matrix = torch.corrcoef(x)
        # print(f"\n\n correlation_matrix is {corr_matrix}\n\n")
        corr = - torch.square(corr_matrix[0][1])
        # print(f"\n\n correlation is {corr}\n\n")
        return corr

    def check_correctness(self, real, pred):
        if len(real) == 0:
            return real, pred
        # print("Real and pred:")
        # print(real,"\n", pred)
        if len(torch.unique(real)) == 1:
            real = real.add(torch.rand(len(real))/1000)
        if len(torch.unique(pred)) == 1:
            pred = pred.add(torch.rand(len(pred))/1000)
        # print("--- filtering --- \nReal and pred:")
        # print(real,"\n", pred)
        return real, pred

