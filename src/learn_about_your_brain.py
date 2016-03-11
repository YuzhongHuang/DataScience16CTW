"""learn_about_your_brain.py
~~~~~~~~~~~~~~

A tool that takes the second and third argument in
the terminal as filepath and image type respectively

There are three possible choices for image type:
cor, sag and tra

An simple assessment for CDR for the input data will
be printed and show an image of the saliency of the
import image
"""

import sys

import pylab
import numpy
import theano
from PIL import Image

import convnet
from convnet import Network
from saliency import saliency
import load_data
from load_data import data_loader

file = sys.argv[1] 
img_type = sys.argv[2]

with open(file, 'r+b') as f:
	with Image.open(f) as im:
		im = numpy.asarray(list(im.getdata()), dtype=theano.config.floatX)

net = Network.load("./CDR_conv/"+img_type.lower()+"_net.pkl")
loader = data_loader.load("./DATA_loader/loader")
saliency_maker = saliency(loader, 2, img_type.upper(), "CDR", net, 10)

print "\nCDR Prediction: ", net.feedforward([im, im, im, im, im, im, im, im, im, im])[0].argmax() / 2.0

brain_mat = saliency_maker.know_your_brain(im)
pylab.imshow(brain_mat, cmap="hot")
pylab.show()
