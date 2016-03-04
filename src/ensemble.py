import convnet
import load_data
import numpy as np
from convnet import Network

training_data, validation_data, test_data = convnet.load_data()

def get_result(net, images):
	lst = []
	for i in range(images.eval().shape[0]):
		lst.append(net.feedforward(images[i], 1))
	return lst

COR = Network.load("", "./trained_model/cor_net.pkl")
# SAG = Network.load("", "./trained_model/sag_net.pkl")
# TRA = Network.load("", "./trained_model/tra_net.pkl")

COR_res = get_result(COR, validation_data[0])
# SAG_res = get_result(SAG)
# TRA_res = get_result(TRA)

# print zip(COR_res, results)	
