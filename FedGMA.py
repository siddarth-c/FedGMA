"""
Name: FedGMA.py
Aim: A federated learning approach
Author: C Siddarth
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

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainx = []
trainy = []

# Load train data
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

def FedAvg(train_x, train_y, test_x, test_y, GD = 'Full'):
    """
    Performs FedAvg to train a global model
    Args: 
        train_x: Training data of shape [#clients, #samples, 28, 28, 3]
        train_y: Training labels of shape [#clients, #samples]
        test_x: Testing data of shape [#clients, #samples, 28, 28, 3]
        test_y: Testing labels of shape [#clients, #samples]
        GD: Type of gradient Descent: Full(Batch), SGD (Default: 'Full')
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
    epochs = 3
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

        print('Communication Round:', cr, ' Loss:', np.round(loss.cpu().detach().numpy(), 4), ' Acc:', np.round(acc.cpu().detach().numpy(), 4))
    
    return test_loss, test_acc

FedAvg_loss, FedAvg_acc = FedAvg(trainx, trainy, testx, testy)

if os.path.exists('Output'):
    shutil.rmtree('Output')
    print('Deleted exisitng *Output* folder')

os.mkdir('Output')
print('Created *Output* folder')

np.save('Output/FedAvg_Acc.npy' , np.array(savgol_filter(FedAvg_acc, 11, 4)))
np.save('Output/FedAvg_Loss.npy' , np.array(savgol_filter(FedAvg_loss, 11, 4)))

def FedGMA(train_x, train_y, test_x, test_y, p_thresh = 0.8, GD = 'Full'):
    """
    Performs FedGMA to train a global model
    Args: 
        train_x: Training data of shape [#clients, #samples, 28, 28, 3]
        train_y: Training labels of shape [#clients, #samples]
        test_x: Testing data of shape [#clients, #samples, 28, 28, 3]
        test_y: Testing labels of shape [#clients, #samples]
        p_thresh: AND mask threshold (default: 0.8)
        GD: Type of gradient Descent: Full(Batch), SGD (Default: 'Full')
    Returns:
        test_loss: Communication round wise test loss
        test_acc: Communication round wise test accuracy
    """
    client_models, global_model = GenerateModels()
 
    for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())
    
    zero_client = MNISTtier().to(device) # A dummy model with all weights set as zero (always)
    sign_counter = MNISTtier().to(device) # A model to aid in AND MASK operation
    
    for p in zero_client.parameters():
        p.data = p.data * 0
        
    comm_rounds = 50
    epochs = 3
    no_clients = 100
    server_lr = 0.0001

    test_loss = []
    test_acc = []

    for cr in range(comm_rounds): # Communication Rounds

        sign_counter.load_state_dict(zero_client.state_dict())

        for ci in range(no_clients): # Iterate through all clients
            
            optimizer = optim.Adam(client_models[ci].parameters(), lr = 0.001)
            criterion = nn.BCEWithLogitsLoss()
            x_600 = train_x[ci]
            y_600 = train_y[ci]

            for e in range(epochs): # Client side epochs

                if GD == 'SGD': # Stochastic Gradient Descent - Very slow
                    for x, y in zip(x_600, y_600):
                        optimizer.zero_grad()
                        pred = client_models[ci](x.reshape((1, 3, 28, 28)))
                        loss = criterion(pred, y)
                        loss.backward()
                        optimizer.step()
                else: # Batch Gradient Descent - Faster than SGD
                    optimizer.zero_grad()
                    pred = client_models[ci](x_600.reshape((600, 3, 28, 28)))
                    loss = criterion(pred, y_600)
                    loss.backward()
                    optimizer.step()


        global_model.load_state_dict(zero_client.state_dict()) 

        # Sum of all client's weights + calculate signs of all graidents
        for ind in range(no_clients):
            for p1, p2, p3 in zip(global_model.parameters(), client_models[ind].parameters(), sign_counter.parameters()):
                p2_grad_sign = torch.sign(p2.grad)
                p3.data += p2_grad_sign
                p1.data = p1.data + p2.data

        for p in global_model.parameters():
            p.data = p.data / no_clients

        # AND MASK
        for ind in range(no_clients):
            for p1, p2, p3 in zip(global_model.parameters(), client_models[ind].parameters(), sign_counter.parameters()):
                p2_mask = 1 * (p2.grad > 0)
                p3_mask = 1 * (p3.data > 0)
                final_mask = torch.logical_and(torch.logical_not(torch.logical_xor(p2_mask, p3_mask)), 1 * (torch.abs(p3.data) > p_thresh * no_clients))
                new_grad = p2.grad * final_mask
            
                p1.data -= server_lr * new_grad/no_clients
        
        # Load all clients with server/global model's weights
        for client_model in client_models:
            client_model.load_state_dict(global_model.state_dict())
        
        # Test time
        pred = global_model(test_x.reshape((10000, 3, 28, 28)))
        loss = criterion(pred, test_y)

        pred = torch.round(F.sigmoid(pred))
        acc = sum(pred == test_y) / len(pred)

        test_loss.append(loss.cpu().detach().numpy())
        test_acc.append(acc.cpu().detach().numpy())

        if cr % 10 == 0:
            print('Communication Round:', cr, ' Loss:', np.round(loss.cpu().detach().numpy(), 4), ' Acc:', np.round(acc.cpu().detach().numpy(), 4))
        
    return test_loss, test_acc


for p in np.arange(0.5, 1, 0.1):
        
    FedGMA_loss, FedGMA_acc = FedGMA(trainx, trainy, testx, testy, p)

    np.save('Output/FedGMA_Acc' + str(p) + '.npy', np.array(savgol_filter(FedGMA_acc, 11, 4)))
    np.save('Output/FedGMA_Loss' + str(p) + '.npy', np.array(savgol_filter(FedGMA_loss, 11, 4)))

    print('Probability threshold', p, 'done \n', '-' * 5)

# # Uncomment to plot the obtained results
# acc = []
# acc_names = []
# loss = []
# loss_names = []

# for fname in glob.glob('Output/*'):
#     ar = np.load(fname)
#     if fname.split('_')[-1][:4] == 'Loss':
#         loss.append(ar)
#         loss_names.append(fname.split('/')[-1].split('n')[0][:-1])
#     else:
#         acc.append(ar)
#         acc_names.append(fname.split('/')[-1].split('n')[0][:-1])

# plt.rcParams["figure.figsize"] = (10,15)

# for a in acc:
#     plt.plot(a)

# plt.xlabel('Communication Rounds')
# plt.ylabel('Test Accuracy')
# plt.legend(acc_names)
# plt.show()

# plt.rcParams["figure.figsize"] = (10,10)

# for l in loss:
#     plt.plot(l)

# plt.xlabel('Communication Rounds')
# plt.ylabel('Test Loss')
# plt.legend(loss_names)
# plt.show()

