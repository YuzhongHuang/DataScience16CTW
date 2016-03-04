import os
from PIL import Image
import numpy
import pandas

def load_image_data(path="../data/TRA_pro/"):
	image_data = []

	files = [fn for fn in os.listdir(path)]
	files.sort()

	for fn in files:
		with open(path+fn, 'r+b') as f:
			with Image.open(f) as im:
				image_data.append(numpy.asarray(list(im.getdata())))
	return numpy.asarray(image_data)

def load_result_data(path="../data/"):
	df = pandas.read_csv(path+"data_summary.csv", sep=',')
	df["CDR"] = df["CDR"].fillna(0.0)
	df["CDR"] = df["CDR"].apply(lambda x:int(2*x))
	# df["CDR"] = df["CDR"].apply(lambda x: x-1 if x == 4 else x)

	return numpy.asarray(df["CDR"].tolist())

def load_all():
	return (
			(load_image_data()[0:120], load_result_data()[0:120]), \
			(load_image_data()[120:240], load_result_data()[120:240]), \
			(load_image_data()[240:-1], load_result_data()[240:-1]), \
			)