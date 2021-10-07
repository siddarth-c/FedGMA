# FedGMA*


This work is inspired by the intuitive approach used in [Gradient-Masked Federated Learning](https://github.com/siddarth-c/FedGMA/blob/main/Extras/GRADIENT-MASKED%20FEDERATED%20OPTIMIZATION.pdf). FedGMA is a modified version of FedAvg that ensures better convergence of server model, especially in the case of NIID data. 

- [Federated learning](#federated-learning)
- [Dataset](#dataset)
- [Results and Observation](#results-and-observation)
- [To Run](#to-run)
- [Citation](#citation)

## Federated learning
Federated learning (also known as collaborative learning) is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. This approach stands in contrast to traditional centralized machine learning techniques where all the local datasets are uploaded to one server, as well as to more classical decentralized approaches which often assume that local data samples are identically distributed. (Wikipedia) <br>

![FL](https://github.com/siddarth-c/FedGMA/blob/main/Extras/FL.png)

<br> FedAvg, or Federated Average, is one of such algorithms introduced by Google in 2017. It is the first ever FL algorithm, and serves as a baseline now for the new methods to beat. For more info on FedAvg, refer to [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf). <br>
FedGMA is an FL algorithm devised by the people at MILA. It uses an AND-Masked gradient update along with parameter averaging to ensure update steps in the direction of the optimal minima across clients. This ensures that the direction of gradient descent is similar to the majority of the participating clients. Find my implementation [here](https://github.com/siddarth-c/FedGMA/blob/main/FedGMA.py)

## Dataset
The authors of the paper use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset to test their proposed work. It contains 60,000 training images and 10,000 testing images. The numbers are color coded with a self-induced noise. In the training set, the numbers below 5 are coloured with red, and the rest with green. This is inverted for the test set inorder to assess the generalization of the model. For more info, refer to the FedGMA paper and section 5.2 in the paper [Invariant Risk Minimization
](https://arxiv.org/pdf/1907.02893.pdf).

## Results and Observation
Multiple models were trained to test 2 different hyper-parameters, namely the client probability threshold and local client epochs.

### Client Probability Threshold
The client probability threshold, P ∈ [0.5, 0.6, 0.7, 0.8, 0.9] was tested and compared with the FedAvg model (E = 3). All these were trained for communication rounds = 50 and local client epochs = 3. The test accuracy was calculated at the end of every communication round and is reported below. Note that the model trained with a probability threshold of 0.7 achieves the maximum accuracy in most of the communication rounds. <br><br>
![Accuracy plot](https://github.com/siddarth-c/FedGMA/blob/main/Extras/Probability.png)

### Local Client Epochs
The local client epochs, E ∈ [1, 3, 5, 7, 9] was tested and compared with the FedAvg model (E = 3). All these were trained for communication rounds = 50 and client probability threshold = 0.7. The test accuracy was calculated at the end of every communication round and is reported below. Note that the model trained for local client epochs of 9 achieves the maximum accuracy in most of the communication rounds.<br><br>
![Accuracy plot](https://github.com/siddarth-c/FedGMA/blob/main/Extras/Epochs.png)
<br>
### The Dip
Notice, there is an initial dip in the performance of all the models before rising. One possible explaination could be the way the model learns. The model could have learnt to classify via 2 different features:
1. Based on colour - Classiying based on colour would be the easiest. Red-> class 0, Green-> class1. But due to the induced errors, this would not be the ideal solution
2. Based on integers - Classying the images based on the pixel locations (the integers itself), which is compartively tough, would be the ideal solution <br>

To speculate, the model could have chosen the easier way at the begininig of classying by colour (local minima), but later realize that this is not the best solution and begins learning it based in the integers itself (global minima). <br>

## Experimental Details
Following are the hyper-parameters used: <br>
1. Optimizer: [Adam](https://arxiv.org/abs/1412.6980)
2. Client learning rate: 0.001
3. Server learning rate: 0.0001

There are a few implementation differences from the paper, optimizers and learning rate to name a few. Though the hyperparameters vary, the core idea is the same. <br> 
Also the graphs were interpolated for the purpose of visualization.
<br><br> 

## To Run
To run the code, follow the following steps:
1. Download the MNIST data from [here](http://yann.lecun.com/exdb/mnist/)
2. Extract the downloaded zip files into a new folder in the root directory titled 'samples'
3. Download this repository as a zip from [here](https://github.com/siddarth-c/FedGMA/archive/refs/heads/main.zip) and extract the files. 
4. Copy all the files in the directory 'working-directory/FedGMA-main/FedGMA-main/' to 'working-directory/'
5. Install the required python packages using the command ```pip install -r requirements.txt```
6. First run the DataDistributor.py file to generate training samples
7. Next run the FedGMA.py to train the model and save the results
   
Your directory after step 4 should resemble the following:
```
working-directory/
    DataDistributor.py
    FedGMA.py
    README.md
    requirements.txt
    Extras/
        Epochs.png
        FL.png
        GRADIENT-MASKED FEDERATED OPTIMIZATION.pdf
        Probability.png
        README.md
    samples/
        t10k-images-idx3-ubyte
        t10k-labels-idx1-ubyte
        train-images-idx3-ubyte
        train-labels-idx1-ubyte
```

## Citation
```
@article{Tenison2021GradientMF,
  title={Gradient Masked Federated Optimization},
  author={Irene Tenison and Sreya Francis and I. Rish},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.10322}
  
  @inproceedings{McMahan2017CommunicationEfficientLO,
  title={Communication-Efficient Learning of Deep Networks from Decentralized Data},
  author={H. B. McMahan and Eider Moore and D. Ramage and S. Hampson and B. A. Y. Arcas},
  booktitle={AISTATS},
  year={2017}
}

@article{Ahuja2020InvariantRM,
  title={Invariant Risk Minimization Games},
  author={Kartik Ahuja and Karthikeyan Shanmugam and K. Varshney and Amit Dhurandhar},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.04692}
}
}


```

\* This is not the official implementation of FedGMA :exclamation:

