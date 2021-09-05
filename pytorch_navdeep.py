# reference url : https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/

import torch
n_input, n_hidden, n_output = 5, 3, 1
print(n_input, n_hidden, n_output)

## initialize tensor for inputs, and outputs 
x = torch.randn((1, n_input))
y = torch.randn((1, n_output))
print(x)
print(y)

## initialize tensor variables for weights 
w1 = torch.randn(n_input, n_hidden) # weight for hidden layer
w2 = torch.randn(n_hidden, n_output) # weight for output layer
print(w1)
print(w2)

## initialize tensor variables for bias terms 
b1 = torch.randn((1, n_hidden)) # bias for hidden layer
b2 = torch.randn((1, n_output)) # bias for output layer
print(b1)
print(b2)

## sigmoid activation function using pytorch
def sigmoid_activation(z):
    return 1 / (1 + torch.exp(-z))
## activation of hidden layer 
z1 = torch.mm(x, w1) + b1
a1 = sigmoid_activation(z1)
## activation (output) of final layer 
z2 = torch.mm(a1, w2) + b2
output = sigmoid_activation(z2)

loss = y - output

## function to calculate the derivative of activation
def sigmoid_delta(x):
  return x * (1 - x)

## compute derivative of error terms
delta_output = sigmoid_delta(output)
delta_hidden = sigmoid_delta(a1)
## backpass the changes to previous layers 
d_outp = loss * delta_output
loss_h = torch.mm(d_outp, w2.t())
d_hidn = loss_h * delta_hidden

learning_rate = 0.1

w2 += torch.mm(a1.t(), d_outp) * learning_rate
w1 += torch.mm(x.t(), d_hidn) * learning_rate
b2 += d_outp.sum() * learning_rate
b1 += d_hidn.sum() * learning_rate

#-----------------------------------------------------------
from torchvision import transforms

_tasks = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    
from torchvision.datasets import MNIST

## Load MNIST Dataset and apply transformations
mnist = MNIST("data", download=True, train=True, transform=_tasks)

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
## create training and validation split 
split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

#print(train_idx)
## create sampler objects using SubsetRandomSampler
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)
#print(tr_sampler)
## create iterator objects for train and valid datasets
trainloader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(mnist, batch_size=256, sampler=val_sampler)
#print(trainloader)

import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)
    def forward(self, x):
        x = self.hidden(x)
        x = F.sigmoid(x)
        x = self.output(x)
        return x
model = Model()

from torch import optim
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)


for epoch in range(1, 11): ## run the model for 10 epochs
    train_loss, valid_loss = [], []
    ## training part 
    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()


#-----------------------------------------------------------
from torchvision import transforms
## load the dataset 
_tasks = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
from torchvision.datasets import CIFAR10
cifar = CIFAR10('data', train=True, download=True, transform=_tasks)
## create training and validation split 
split = int(0.8 * len(cifar))
index_list = list(range(len(cifar)))
train_idx, valid_idx = index_list[:split], index_list[split:]
## create training and validation sampler objects
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)
## create iterator objects for train and valid datasets
trainloader = DataLoader(cifar, batch_size=512, sampler=tr_sampler)
validloader = DataLoader(cifar, batch_size=512, sampler=val_sampler)

import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        ## define the layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 1024) ## reshaping 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = Model()

import torch.optim as optim
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9, nesterov = True)
## run for 30 Epochs
for epoch in range(1, 2):
    train_loss, valid_loss = [], []
    ## training part 
    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item()) 
        
    ## evaluation part 
    model.eval()
    for data, target in validloader:
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
    print ("Epoch:", epoch)
        
## dataloader for validation dataset 
dataiter = iter(validloader)
data, labels = dataiter.next()
output = model(data)
print(output)
output.shape
import numpy as np
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
print ("Actual:", labels)
print ("Predicted:", preds)


































