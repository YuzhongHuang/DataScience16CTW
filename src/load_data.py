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

		self.test_size = 100
		self.val_size = 100
		self.train_size = 216

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

	def load_for_learn(self, img_type, result):
		test = self.train_size
		val = self.train_size + self.val_size

		return (
				(self.img_data[img_type][val:-1], 		self.res_data[img_type][result][val:-1]), \
				(self.img_data[img_type][0:test], 		self.res_data[img_type][result][0:test]), \
				(self.img_data[img_type][test:val],	self.res_data[img_type][result][test:val]), \
				)

data_loader().save("./DATA_loader/loader")