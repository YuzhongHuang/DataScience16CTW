import convnet
from convnet import Network

import load_data
from load_data import data_loader

from saliency import saliency

net = Network.load("./CDR_conv/sag_net.pkl")
loader = data_loader.load("./DATA_loader/loader")
saliency_maker = saliency(loader, 2, "SAG", "CDR", net)

print saliency_maker.get_heap_mat()
saliency_maker.save("./CDR_saliency/sag_train.pkl")