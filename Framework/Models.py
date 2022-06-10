from torch import nn

# Building a Neural Network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            #nn.ReLU(),
            #nn.Linear(512, 2),
            #nn.ReLU(),
            #nn.Linear(128, 2),
            #nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# In[11]:


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
        return logits


# In[12]:


# Building a Neural Network architecture
class Br_Net(nn.Module):
    def __init__(self):
        super(Br_Net, self).__init__()
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
            nn.Linear(32,2),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
