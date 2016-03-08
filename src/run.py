import convnet
from convnet import Network
from convnet import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

import load_data
from load_data import data_loader

loader = data_loader.load("./DATA_loader/loader")

training_data, validation_data, test_data = convnet.load_data(loader, "COR")
# mini_batch_size = 10

# net = Network([
#         ConvPoolLayer(image_shape=(mini_batch_size, 1, 44, 44), 
#                       filter_shape=(20, 1, 5, 5), 
#                       poolsize=(2, 2)),
#         ConvPoolLayer(image_shape=(mini_batch_size, 20, 20, 20), 
#                       filter_shape=(40, 20, 5, 5), 
#                       poolsize=(2, 2)),
#         FullyConnectedLayer(n_in=40*8*8, n_out=100),
#         SoftmaxLayer(n_in=100, n_out=4)], mini_batch_size)

# net.SGD(training_data, 30, mini_batch_size, 1, 
#             validation_data, test_data, lmbda=0.1)

# net.save("./CDR_conv/cor.pkl")

# net = Network.load("./CDR_conv/cor.pkl")
# net.feedforward(loader.img_data["COR"][300:310])
