"""
Name: DataDistributor.py
Aim: Convert MNIST data into binary classification data
Author: Siddarth C
Date: September, 2021
"""

from mnist import MNIST
import numpy as np
import random
import os
import shutil

random.seed(100)


if os.path.exists('ClientData'):
    shutil.rmtree('ClientData')
    print('Deleted exisitng *ClientData* folder')

os.mkdir('ClientData')
print('Created *ClientData* folder')

mndata = MNIST('samples')

print('Loading Training Data...')
images, labels = mndata.load_training()

print('Processing Training Data...')
images = np.array(images).reshape((60000, 28, 28))
labels = np.array(labels)

inds = labels.argsort()
images = images[inds]
labels = labels[inds]

sorted = [images[i*6000:(i+1)*6000] for i in range(10)]

trainx = []
trainy = []

mnist_200 = [i for i in range(10) for j in range(20)]

mnist_10_20_300 = []

for i in range(10):
    dummy = [j for j in range(6000)]
    random.shuffle(dummy)
    mnist_10_20_300.append([dummy[j*300:(j+1)*300] for j in range(20)])

trainx = []
trainy = []
client_id = 0
safe_value = 0
while len(mnist_200) > 0:
    n1 = random.randint(0, len(mnist_200) - 1)
    n2 = random.randint(0, len(mnist_200) - 1)
    no1 = mnist_200[n1]
    no2 = mnist_200[n2]
    safe_value += 1
    if safe_value > 10e6:
        print('Error. Please run the program again!')
        exit()
    if no1 != no2:
        client_id += 1
        indiv_client_x = []
        indiv_client_y = []
        no1_index = mnist_10_20_300[no1][0]
        no2_index = mnist_10_20_300[no2][0]
        mnist_10_20_300[no1] = mnist_10_20_300[no1][1:]
        mnist_10_20_300[no2] = mnist_10_20_300[no2][1:]
        client_prob = random.randint(100,200)/1000
        for no, no_index in zip([no1, no2], [no1_index, no2_index]):
            z = 0
            for no_i in no_index:
                img = sorted[no][no_i]
                data_prob = random.random()
                red_version = np.stack((img, np.zeros_like(img), np.zeros_like(img)), axis = 2)
                green_version = np.stack((np.zeros_like(img), img, np.zeros_like(img)), axis = 2)
                if (data_prob > client_prob and no < 5) or (data_prob < client_prob and no > 5):
                    indiv_client_x.append(red_version)
                else:
                    indiv_client_x.append(green_version)                    
                indiv_client_y.append(1*(no>4))

        os.mkdir('ClientData/Client' + str(client_id))
        np.save('ClientData/Client' + str(client_id) + '/y.npy', np.stack(indiv_client_y))
        np.save('ClientData/Client' + str(client_id) + '/x.npy', np.stack(indiv_client_x))
        del mnist_200[n1]
        if n2 > n1:
            del mnist_200[n2 - 1]
        else:
            del mnist_200[n2]

print('Distributed training data among clients!')

print()

if os.path.exists('TestData'):
    shutil.rmtree('TestData')
    print('Deleted exisitng *TestData* folder')

os.mkdir('TestData')
print('Created *TestData* folder')

print('Loading Test Data...')
images, labels = mndata.load_testing()

print('Processing Test Data...')
images = np.array(images).reshape((10000, 28, 28))
labels = np.array(labels)

testx = []
testy = []

for i, l in zip(images, labels):
    red_version = np.stack((i, np.zeros_like(i), np.zeros_like(i)), axis = 2)
    green_version = np.stack((np.zeros_like(i), i, np.zeros_like(i)), axis = 2)
    data_prob = random.randint(100,200)/1000
    if (data_prob > 0.9 and l > 5) or (data_prob < 0.9 and l < 5):
        testx.append(green_version)
    else:
        testx.append(red_version)
    testy.append(1*(l>4))

np.save('TestData/x.npy', testx)
np.save('TestData/y.npy', testy)

print('Test data saved!')
