"""
Name: FedGMA.py
Aim: To test the hyper-parameters used in GRADIENT-MASKED FEDERATED OPTIMIZATION
Author: Siddarth C
Date: September, 2021
"""

# Import required libraries
import os
import shutil
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Setting random seed for reproducability
import random
random.seed(7)

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists('Output'):
    shutil.rmtree('Output')
    print('Deleted exisitng *Output* folder')

os.mkdir('Output')
os.mkdir('Output/P')
os.mkdir('Output/E')
print('Created *Output* and sub folders')

# Load train data
trainx = []
trainy = []

# Load test data
for folder in glob.glob('ClientData/*'):
    x = np.load(folder + '/x.npy')
    y = np.load(folder + '/y.npy')
    trainx.append(torch.FloatTensor(x))
    trainy.append(torch.FloatTensor(y))

trainx = torch.stack(trainx).to(device)
trainy = torch.stack(trainy).to(device)

# Load test data
testx = torch.FloatTensor(np.load('TestData/x.npy')).to(device)
testy = torch.FloatTensor(np.load('TestData/y.npy')).to(device)

# Define the classifier - MNISTtier
class MNISTtier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

def GenerateModels():
    """
    Returns client and global model
    Args: 
        None
    Returns:
        100 client models, Global model
    """
    client_models = [MNISTtier().to(device) for i in range(100)]
    global_model = MNISTtier().to(device)

    return client_models, global_model

def FedAvg(train_x, train_y, test_x, test_y, GD = 'Full', epochs = 3):
    """
    Performs FedAvg to train a global model
    Args: 
        train_x: Training data of shape [#clients, #samples, 28, 28, 3]
        train_y: Training labels of shape [#clients, #samples]
        test_x: Testing data of shape [#clients, #samples, 28, 28, 3]
        test_y: Testing labels of shape [#clients, #samples]
        GD: Type of gradient Descent: Full(Batch), SGD (Default: 'Full')
        epochs: Number of local client epochs (Default: 3)
    Returns:
        test_loss: Communication round wise test loss
        test_acc: Communication round wise test accuracy
    """
    client_models, global_model = GenerateModels()
 
    for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())
 
    zero_client = MNISTtier().to(device)

    for p in zero_client.parameters():
        p.data = p.data * 0
        
    comm_rounds = 50
    no_clients = 100

    test_loss = []
    test_acc = []

    for cr in range(comm_rounds):
        for ci in range(no_clients):
            
            optimizer = optim.Adam(client_models[ci].parameters(), lr = 0.001)
            criterion = nn.BCEWithLogitsLoss()
            x_600 = train_x[ci]
            y_600 = train_y[ci]

            for e in range(epochs):

                if GD == 'SGD':
                    for x, y in zip(x_600, y_600):
                        optimizer.zero_grad()
                        pred = client_models[ci](x.reshape((1, 3, 28, 28)))
                        loss = criterion(pred, y)
                        loss.backward()
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    pred = client_models[ci](x_600.reshape((600, 3, 28, 28)))
                    loss = criterion(pred, y_600)
                    loss.backward()
                    optimizer.step()


        global_model.load_state_dict(zero_client.state_dict()) 

        for ind in range(no_clients):
            for p1, p2 in zip(global_model.parameters(), client_models[ind].parameters()):
                p1.data = p1.data + p2.data

        for p in global_model.parameters():
            p.data = p.data / no_clients

        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())
        
        pred = global_model(test_x.reshape((10000, 3, 28, 28)))
        loss = criterion(pred, test_y)

        pred = torch.round(F.sigmoid(pred))
        acc = sum(pred == test_y) / len(pred)

        test_loss.append(loss.cpu().detach().numpy())
        test_acc.append(acc.cpu().detach().numpy())

        if cr % 20 == 0:
            print('Communication Round:', cr, ' Loss:', np.round(loss.cpu().detach().numpy(), 4), ' Acc:', np.round(acc.cpu().detach().numpy(), 4))
    
    return test_loss, test_acc

FedAvg_loss, FedAvg_acc = FedAvg(trainx, trainy, testx, testy)


np.save('Output/P/FedAvg_Acc.npy' , np.array(savgol_filter(FedAvg_acc, 9, 4)))
np.save('Output/P/FedAvg_Loss.npy' , np.array(savgol_filter(FedAvg_loss, 9, 4)))

np.save('Output/E/FedAvg_Acc_3.npy' , np.array(savgol_filter(FedAvg_acc, 9, 4)))
np.save('Output/E/FedAvg_Loss_3.npy' , np.array(savgol_filter(FedAvg_loss, 9, 4)))

def FedGMA(train_x, train_y, test_x, test_y, p_thresh = 0.8, GD = 'Full', epochs = 3):
    """
    Performs FedGMA to train a global model
    Args: 
        train_x: Training data of shape [#clients, #samples, 28, 28, 3]
        train_y: Training labels of shape [#clients, #samples]
        test_x: Testing data of shape [#clients, #samples, 28, 28, 3]
        test_y: Testing labels of shape [#clients, #samples]
        p_thresh: AND mask threshold (default: 0.8)
        GD: Type of gradient Descent: Full(Batch), SGD (Default: 'Full')
        epochs: Number of local client epochs (Default: 3)
    Returns:
        test_loss: Communication round wise test loss
        test_acc: Communication round wise test accuracy
    """
    client_models, global_model = GenerateModels()
 
    for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())

    
    zero_client = MNISTtier().to(device)
    sign_counter = MNISTtier().to(device)
    
    for p in zero_client.parameters():
        p.data = p.data * 0
        
    comm_rounds = 50
    no_clients = 100
    server_lr = 0.0001

    test_loss = []
    test_acc = []

    for cr in range(comm_rounds):

        sign_counter.load_state_dict(zero_client.state_dict())

        for ci in range(no_clients):
            
            optimizer = optim.Adam(client_models[ci].parameters(), lr = 0.001)
            criterion = nn.BCEWithLogitsLoss()
            x_600 = train_x[ci]
            y_600 = train_y[ci]

            for e in range(epochs):

                if GD == 'SGD':
                    for x, y in zip(x_600, y_600):
                        optimizer.zero_grad()
                        pred = client_models[ci](x.reshape((1, 3, 28, 28)))
                        loss = criterion(pred, y)
                        loss.backward()
                        optimizer.step()
                else:
                    optimizer.zero_grad()
                    pred = client_models[ci](x_600.reshape((600, 3, 28, 28)))
                    loss = criterion(pred, y_600)
                    loss.backward()
                    optimizer.step()


        global_model.load_state_dict(zero_client.state_dict()) 

        for ind in range(no_clients):
            for p1, p2, p3 in zip(global_model.parameters(), client_models[ind].parameters(), sign_counter.parameters()):
                p2_grad_sign = torch.sign(p2.grad)
                p3.data += p2_grad_sign
                p1.data = p1.data + p2.data

        for p in global_model.parameters():
            p.data = p.data / no_clients

           
        for ind in range(no_clients):
            for p1, p2, p3 in zip(global_model.parameters(), client_models[ind].parameters(), sign_counter.parameters()):
                p2_mask = 1 * (p2.grad > 0)
                p3_mask = 1 * (p3.data > 0)
                final_mask = torch.logical_and(torch.logical_not(torch.logical_xor(p2_mask, p3_mask)), 1 * (torch.abs(p3.data) > p_thresh * no_clients))
                new_grad = p2.grad * final_mask
            
                p1.data -= (server_lr * new_grad/no_clients)
                
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())
        
        pred = global_model(test_x.reshape((10000, 3, 28, 28)))
        loss = criterion(pred, test_y)

        pred = torch.round(F.sigmoid(pred))
        acc = sum(pred == test_y) / len(pred)

        test_loss.append(loss.cpu().detach().numpy())
        test_acc.append(acc.cpu().detach().numpy())


        if cr % 20 == 0:
            print('Communication Round:', cr, ' Loss:', np.round(loss.cpu().detach().numpy(), 4), ' Acc:', np.round(acc.cpu().detach().numpy(), 4))
        
    return test_loss, test_acc

print('FedGMA - Testing of hyperparmeter - P - Threshold of number of client gradients to be consistent')

for p in np.arange(0.5, 1, 0.1):
    
    p = np.round(p, 1)
    FedGMA_loss, FedGMA_acc = FedGMA(trainx, trainy, testx, testy, p_thresh = p)
    np.save('Output/P/FedGMA_Acc_' + str(p) + '.npy', np.array(savgol_filter(FedGMA_acc, 9, 4)))
    np.save('Output/P/FedGMA_Loss_' + str(p) + '.npy', np.array(savgol_filter(FedGMA_loss, 9, 4)))

    print('Probability threshold', p, 'done \n', '-' * 5)

print('')
print('FedGMA - Testing of hyperparmeter - Local Client Epochs')

for epochs in [1, 3, 5, 7, 9]:
    
    FedGMA_loss, FedGMA_acc = FedGMA(trainx, trainy, testx, testy, p_thresh = 0.7, epochs = epochs)

    np.save('Output/E/FedGMA_Acc_' + str(epochs) + '.npy', np.array(savgol_filter(FedGMA_acc, 9, 4)))
    np.save('Output/E/FedGMA_Loss_' + str(epochs) + '.npy', np.array(savgol_filter(FedGMA_loss, 9, 4)))

    print('Local Epochs', epochs, 'done \n', '-' * 5)


acc = []
acc_names = []
loss = []
loss_names = []

hyper_parameter = 'P' # or 'E'

# Please do modify the following part if the graphs are not plotted
for fname in glob.glob('Output/' + hyper_parameter + '/*.npy'):
    ar = np.load(fname)
    if fname.split('/')[-1].split('_')[1] == 'Loss':
        loss.append(ar)
        loss_names.append(fname.split('\\')[-1].split('.npy')[0])
    else:
        acc.append(ar)
        acc_names.append(fname.split('\\')[-1].split('.npy')[0])

plt.rcParams["figure.figsize"] = (15,15)

for a in acc:
    plt.plot(a)

plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy')
plt.legend(acc_names)
plt.show()