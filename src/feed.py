"""feed.py
~~~~~~~~~~~~~~

A script that feeds data into the model given a trained network 

The example code is going to load a neural network of 
CONV-POOL-CONV-POOL-FULL-SOFTMAX
The mini batch size is 10 'COR' image for each epoch of 100 epoch.
The learning step size here is 0.03

The test image is the first 10 images

Details for parameters specific to each layer can find in the given
example in the file "train.py"

Note that the size of input image list should be same as the mini 
batch size of training network. The feeding takes about several 
seconds. After the script finishes running, it will print out the 
results of given image list in the form as illustrated: 

[probablity for being CDR 0, probablity for being CDR 0.5, probablity for being CDR 1]

The actual results will also be printed for reference.

"""

import convnet
from convnet import Network
from convnet import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

import load_data
from load_data import data_loader
from PIL import Image

loader = data_loader.load("./DATA_loader/loader")
net = Network.load("./CDR_conv/cor_net.pkl")

print net.feedforward(loader.img_data["COR"][0:10])
print loader.res_data["CDR"][0:10]
