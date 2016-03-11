"""make_saliency_map.py
~~~~~~~~~~~~~~

A script that makes a saliency map to show which
part is the trained model looking at.

In the example code, the first 10 'SAG' images will be 
scanned and overlapped to generate a saliency map.

The saliency map object will be stored under the path
"./CDR_saliency/sag_train.pkl"

Details of the parameters are explained in the file
'saliency.py'

"""

import convnet
from convnet import Network

import load_data
from load_data import data_loader

from saliency import saliency

net = Network.load("./CDR_conv/sag_net.pkl")
loader = data_loader.load("./DATA_loader/loader")
saliency_maker = saliency(loader, 2, "SAG", "CDR", net, 10)

saliency_maker.get_heap_mat()
saliency_maker.save("./CDR_saliency/sag_train.pkl")