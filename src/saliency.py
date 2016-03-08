import pickle
import numpy as np
import convnet
from convnet import Network

class saliency:
	def __init__(self, loader, box_size, img_type, result, net):
		self.size = box_size
		self.imgs = loader.img_data[img_type]
		self.result = loader.res_data[img_type][result]

		self.conv = {"COR":2560, "SAG":3200, "TRA":3200}
		self.conv_shape = self.conv[img_type]

		self.shape = {"COR":(44, 44), "SAG":(52, 44), "TRA":(44, 52)}	
		self.img_shape = self.shape[img_type]

		self.net = net
		self.mat_size = (self.img_shape[1]-self.size+1, self.img_shape[0]-self.size+1)

	def save(self, filename):
		with open(filename, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(filename):
		with open(filename, 'rb') as inpt:
			return pickle.load(inpt)

	def get_heap_mat(self):
		mat = np.zeros(self.mat_size)
		vfunc = np.vectorize(lambda x: 1/x)
		for i in range(len(self.imgs[0:1])):
			mat += self.get_img_mat(self.imgs[0:1][i], self.result[i])

		for i in range(len(mat)):
			mat[i] = vfunc(mat[i])

		self.mat = mat
		return mat

	def get_img_mat(self, img, result):
		mat = np.zeros(self.mat_size)
		for i in range(self.mat_size[0]):
			for j in range(self.mat_size[1]):
				print "slow " + str(i) + " " + str(j)
				data = self.black_box(img, (i,j))
				res = self.net.feedforward(data, self.conv_shape)
				mat[i][j] = res[0][result]
		return mat

	def black_box(self, img, box_pos):
		for i in get_index(self.img_shape, self.size, box_pos):
			img[i] = 0
		return img

def get_index(img_shape, box_size, box_pos): 
	lst = []
	for i in range(box_size):
		for j in range(box_size):
			index = (box_pos[0]+i)*img_shape[1] + (box_pos[1]+j)
			lst.append(index)
	return lst