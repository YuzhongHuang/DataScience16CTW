import os
from PIL import Image
import numpy
import pandas

def load_image_data(path="../data/COR_pro/"):
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

	return numpy.asarray(df["CDR"].tolist())

def get_id(s):
	return int(s[5:9])

def merge_data(image, cdr):
	return zip(image, cdr)

print len(load_result_data())
print len(load_image_data())
# merge_data(load_image_data(), load_result_data())