"""train.py
~~~~~~~~~~~~~~

A script that trains the model given network structure and parameters. 

The example code is going to train a neural network of 
CONV-POOL-CONV-POOL-FULL-SOFTMAX
The mini batch size is 10 'COR' image for each epoch of 100 epoch.
The learning step size here is 0.03

Interpretation for parameters specific to each layer can find in
file "convnet.py"

The trainning takes about 10 minutes. After the trainning is done, the
network will be saved in the path "./CDR_conv/test.pkl"

The model gives about 82 percents in the test data

"""

import convnet
from convnet import Network
from convnet import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

import load_data
from load_data import data_loader

loader = data_loader.load("./DATA_loader/loader")

training_data, validation_data, test_data = convnet.load_data("COR")
mini_batch_size = 10

net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 44, 44), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2)
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 20, 20), 
                      filter_shape=(40, 20, 5, 5), 
                      poolsize=(2, 2)
                      activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*8*8, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=3)], mini_batch_size)

net.SGD(training_data, 100, mini_batch_size, 0.01, 
            validation_data, test_data, lmbda=0.1)

net.save("./CDR_conv/test.pkl")