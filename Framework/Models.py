from torch import nn
from Framework.Layers import GradientReversal

# Building a Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
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
class Br_Net(nn.Module):
    def __init__(self, n_classes=2):
        super(Br_Net, self).__init__()
        self.alpha = None
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
        return logits, None

    def get_name(self):
        return "BrNet"

class Br_Net_CF_free(nn.Module):
    def __init__(self, alpha, n_classes=2):
        super(Br_Net_CF_free, self).__init__()
        self.alpha = alpha
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
            nn.Linear(32,n_classes)
        )

        self.domain_predictor = nn.Sequential(
            nn.Linear(32,n_classes)
        )

    def forward(self, x):
        features = self.linear_relu_stack(x)
        reverse_features = GradientReversal.apply(features, self.alpha)

        class_features = self.class_predictor(features)
        domain_features = self.domain_predictor(reverse_features)
        return class_features, domain_features

    def get_name(self):
        return "BrNet_CF_free"

class Br_Net_CF_free_conditioned(nn.Module):
    def __init__(self, alpha, n_classes=2):
        super(Br_Net_CF_free_conditioned, self).__init__()
        self.alpha = alpha
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
            nn.Linear(32,n_classes)
        )

        self.domain_predictor = nn.Sequential(
            nn.Linear(32,n_classes)
        )

    def forward(self, x):
        features = self.linear_relu_stack(x)
        reverse_features = GradientReversal.apply(features, self.alpha)

        class_features = self.class_predictor(features)
        domain_features = self.domain_predictor(reverse_features)
        return class_features, domain_features

    def get_name(self):
        return "BrNet_CF_free_conditioned"

class Br_Net_DANN(nn.Module):
    def __init__(self, alpha, n_classes=2):
        super(Br_Net_DANN, self).__init__()
        self.alpha = alpha
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
            nn.Linear(32,n_classes)
        )

        self.domain_predictor = nn.Sequential(
            nn.Linear(32,n_classes)
        )

    def forward(self, x):
        features = self.linear_relu_stack(x)
        reverse_features = GradientReversal.apply(features, self.alpha)

        class_features = self.class_predictor(features)
        domain_features = self.domain_predictor(reverse_features)
        return class_features, domain_features

    def get_name(self):
        return "BrNet_DANN"

class SimpleConv_DANN(nn.Module):
    def __init__(self, alpha):
        super(SimpleConv_DANN, self).__init__()
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
