import pickle
import numpy as np
import convnet
import theano
from convnet import Network
from PIL import Image

class saliency:
	def __init__(self, loader, box_size, img_type, result, net):
		self.size = box_size
		self.imgs = loader.img_data[img_type]
		self.result = loader.res_data[result]

		self.conv = {"COR":2560, "SAG":3200, "TRA":3200}
		self.conv_shape = self.conv[img_type]

		self.shape = {"COR":(44, 44), "SAG":(44, 52), "TRA":(52, 44)}	
		self.img_shape = self.shape[img_type]

		self.net = net
		self.mat_size = (self.img_shape[0]-self.size+1, self.img_shape[1]-self.size+1)

	def save(self, filename):
		with open(filename, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(filename):
		with open(filename, 'rb') as inpt:
			return pickle.load(inpt)

	def get_heap_mat(self):
		mat = np.zeros(self.mat_size)
		for i in range(len(self.imgs[0:120])):
			print "got ", i
			cur_mat = self.get_img_mat(self.imgs[0:120][i], self.result[i])
			mat += self.process(cur_mat, self.imgs[0:120][i])
		self.mat = mat
		# Image.fromarray(self.imgs[0:1][0].reshape((44,52))).convert('L').save("test", "JPEG")
		return mat

	def process(self, img_mat, img):
		mat = np.ones(self.mat_size)
		actual = self.net.feedforward([img for i in range(10)])[0]
		js = [0, 10, 20, 30, 40, 41]
		for i in range(self.mat_size[0]):
			for j in js:
				res = self.net.feedforward(img_mat[i][j:j+10])
				mat[i][j:j+10] = self.compare(res, actual)
		return mat

	def compare(self, results, actual):
		vfunc = np.vectorize(lambda x: x*x)
		variations = np.zeros(10)
		for i in range(len(results)):
			variations[i] = np.sum(vfunc(results[i]-actual))
		return variations

	def get_img_mat(self, img, result):
		mat = np.zeros(self.mat_size + (self.img_shape[0]*self.img_shape[1],), dtype=theano.config.floatX)
		for i in range(self.mat_size[0]):
			for j in range(self.mat_size[1]):
				mat[i][j] = self.black_box(img, (i,j))
		return mat

	def black_box(self, img, box_pos):
		cur_img = np.copy(img)
		for i in get_index(self.img_shape, self.size, box_pos):
			cur_img[i] = 0
		return cur_img

def get_index(img_shape, box_size, box_pos): 
	lst = []
	for i in range(box_size):
		for j in range(box_size):
			index = (box_pos[0]+i)*img_shape[1] + (box_pos[1]+j)
			lst.append(index)
	return lst