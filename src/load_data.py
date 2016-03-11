"""load_data.py
~~~~~~~~~~~~~~

A local data loader program that grab image data and its corresponding
result data.

To make the loader more efficient, the program allows user to save 
and load the loader object through the use of pickle.

All the images are stored in its attribute "img_data", which is a library
indexed by image types.

All the corresponding datas are stored in its attribute "res_data", which 
is a library indexed by result types.

The another main function of this tool is load_for_learn(), which seperates
data into three sections representing

"""

import os
from PIL import Image
import theano
import numpy
import pandas
import pickle

class data_loader:

	def __init__(self, paths={"COR":"../data/COR_pro/", "SAG":"../data/SAG_pro/", "TRA":"../data/TRA_pro/"}, results=["CDR"]):
		self.paths = paths
		self.results = results

		self.img_data = {}
		self.res_data = {}

		self.train_size = 160
		self.val_size = 80

		for img_type in paths:
			self.img_data[img_type] = self.load_image_data(paths[img_type])
		for result in results:	
			self.res_data[result] = self.load_result_data(result)

	def save(self, filename):
		with open(filename, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(filename):
		with open(filename, 'rb') as inpt:
			return pickle.load(inpt)

	def load_for_learn(self, img_type, result):
		train = self.train_size
		val = self.train_size + self.val_size

		return (
				(self.img_data[img_type][0:train], 		self.res_data[result][0:train]), \
				(self.img_data[img_type][train:val],	self.res_data[result][train:val]), \
				(self.img_data[img_type][val:-1], 		self.res_data[result][val:-1]), \
				)

	#### Helper function for __init__ to grab image and result data
	def load_image_data(self, path):
		image_data = []

		files = [fn for fn in os.listdir(path)]
		files.sort()

		for fn in files:
			with open(path+fn, 'r+b') as f:
				with Image.open(f) as im:
					image_data.append(numpy.asarray(list(im.getdata()), dtype=theano.config.floatX))
		return numpy.asarray(image_data)

	def load_result_data(self, column, path="../data/processed_data_summary.csv"):
		df = pandas.read_csv(path, sep=',')
		lst = df[column].tolist()
		return numpy.asarray(lst)

# create a loader if not already exists
data_loader().save("./DATA_loader/loader")


