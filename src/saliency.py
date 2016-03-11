"""saliency.py
~~~~~~~~~~~~~~

A saliency map generation program for trained models. A saliency map
visualize the how important each part is in the image in predicting 
the result of the model.

A saliency map is created by moving a black box over the image
and generates a series of black-boxed image and feed this series 
of images to trained model. And computes the square errors between
result from original image and those of the black-boxed images.
And then puts the square errors to a matrix.

save() and load() are also included since the object takes time
to get trained.

"""


import pickle
import numpy as np
import convnet
import theano
from convnet import Network
from PIL import Image

class saliency:
	def __init__(self, loader, box_size, img_type, result, net, num_imgs):
		""" The saliency object takes a data loader, the black box's size,
		image's type, the neural network and number of images create a saliency
		map object

		"""

		self.size = box_size
		self.imgs = loader.img_data[img_type]
		self.result = loader.res_data[result]
		self.num_imgs = num_imgs

		self.conv = {"COR":2560, "SAG":3200, "TRA":3200}
		self.conv_shape = self.conv[img_type]

		self.js = {"COR":[0, 10, 20, 30, 33], "SAG":[0, 10, 20, 30, 40, 41], "TRA":[0, 10, 20, 30, 33]}
		self.img_js = self.js[img_type]

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

	def know_your_brain(self, img):
		cur_mat = self.get_img_mat(img)
		mat = self.process(cur_mat, img)
		return mat

	def get_heap_mat(self):
		"""Generate the final matrix of all the square errors and store it to
		self.mat

		"""
		mat = np.zeros(self.mat_size)

		for i in range(len(self.imgs[0:num_imgs])): # loop through all the images
			print "got ", i # print out progress while waiting
			# generate a matrix of all the black-boxed images array
			cur_mat = self.get_img_mat(self.imgs[0:num_imgs][i])
			# process and overlap the square errors of the current image to others 
			mat += self.process(cur_mat, self.imgs[0:num_imgs][i])
		self.mat = mat
		# Image.fromarray(self.imgs[0:1][0].reshape((44,52))).convert('L').save("test", "JPEG")
		return mat

	def process(self, img_mat, img):
		"""Takes a black-boxed image and the original image to calculate 
		and returns the square error

		"""
		mat = np.ones(self.mat_size)
		actual = self.net.feedforward([img for i in range(10)])[0]
		js = self.img_js
		for i in range(self.mat_size[0]):
			for j in js:
				res = self.net.feedforward(img_mat[i][j:j+10])
				mat[i][j:j+10] = self.compare(res, actual)
		return mat

	def compare(self, results, actual):
		"""Given two arrays and returns the sum of their square differences

		"""
		vfunc = np.vectorize(lambda x: x*x)
		variations = np.zeros(10)
		for i in range(len(results)):
			variations[i] = np.sum(vfunc(results[i]-actual))
		return variations

	def get_img_mat(self, img):
		"""Generate a matrix of all the black-boxed images in the form
		of numpy array

		"""
		mat = np.zeros(self.mat_size + (self.img_shape[0]*self.img_shape[1],), dtype=theano.config.floatX)
		for i in range(self.mat_size[0]):
			for j in range(self.mat_size[1]):
				mat[i][j] = self.black_box(img, (i,j))
		return mat

	def black_box(self, img, box_pos):
		"""Given an image and a black box position, returns the black-
		boxed image in the form of an numpy array

		"""
		cur_img = np.copy(img)
		for i in get_index(self.img_shape, self.size, box_pos):
			cur_img[i] = 0
		return cur_img

def get_index(img_shape, box_size, box_pos): 
	"""Helper function to convert indexes in 2D array to 1D array

	"""
	lst = []
	for i in range(box_size):
		for j in range(box_size):
			index = (box_pos[0]+i)*img_shape[1] + (box_pos[1]+j)
			lst.append(index)
	return lst