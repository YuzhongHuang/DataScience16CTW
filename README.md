# MRI Based Clinical Dementia Rating Predictor

## Dataset
Cross sectional MRI brain image data from Open Access Series of Imaging Studies ([OASIS](http://www.oasis-brains.org/))

## Team
Yuzhong Huang and Wilson Tang

## Project Description
Healthcare is an ever evolving field and medical imaging is an essential tool for diagnosing and assessing various conditions from neuro-degenerative diseases like Alzheimer’s to characterizing overall health based on factors like brain volume. 

To understand how MRI images and CDR related and explore more about image processing and deep learning, we built a 6-layer convolutional neural network using theano for a MRI based CDR prediction model.

## Project Details
We started out by downsizing the image data since the neural network structure we are referencing typically works better with smaller images(we used data from OASIS). Specifically, we change "COR" images from 176*176 to 44*44; "SAG" images from 208*176 to 52*44; "TRA" images from 176*208 to 44*52. After data processing, we built a 6-layer convolutional neural network in a structure of "CONV-POOL-CONV-POOL-FULLYCONNECTED-SOFTMAX". We used stochastic gradient decent and back propagation in training our model. After 100 epochs of training, we got about 82% score in the test data.

Note that during our data processing, we also convert all the 'NAN' CDR value to 0 since most of the 'NAN' subject's age tends to be relatively young and we assume that their CDR is 0. That might cause some errors in our model though. However, since we only have 416 data entries and a large portion of them are 'NAN' in CDR, we finally decide to use the data anyway.

To visualize our trainning, we used a method called saliency map. We basically used a black box to run through an image to create a series of black-boxed images. We then fed these images to our model and get their prediction probablities. And then compute the square differences between the original image and black-boxed images. Since these images have black box over different places, by computing the square differences, we can see how much the model is dependent on the part we are covering with the black box. Then we just placed the square errors into a matrix of the same size as the original image. We can use this matrix to visualize the matrix of square errors to see how important each part of the image.

Finally, we just put everything together and create a simple tool to predict the CDR of input MRI image and shows a saliency map of the image.

## Result Example
![examples](https://lh3.googleusercontent.com/ndDA5vOmuOu5qHQBh4UjaSBq33VNblNiDI5Qcg74mwE5--N4_04KI-mxL--ZkF4TFnps=s0 "brains.jpg")

## How To

- All the MRI image data in the form of GIF and result data in CSV are under folder 'data'

- 'res' folder contains the example results of our project. Specifically, saliency maps for different type of MRI images.

- All the project description files are under directory 'files'. Including project proporsal, reflection etc.

- Each source files under 'src' is well-documented and relatively self-explanatory. Folder 'data_processing' and 'image_processing' contains file that process the raw data to be useful for our model. 'CDR_saliency', 'CDR_conv', 'DATA_loader' contains saliency map objects, trained network objects and data loader object respectively. 'testdata' includes image data for 'learn_about_your_brain.py'.

- You may want to use 'train.py' to train your own neural network; 'feed.py' to feed data into your trained network or just use the example network we provided; 'make_saliency_map.py' to make a saliency map to make your own saliency map to visualize features your network is paying attention to. You can bring up the ipython notebook 'visualize.ipynb' to see the saliency map. Finally, you can use 'learn_about_your_brain.py' to see our model's CDR prediction for the given MRI image and its feature interests. Details for how to use each files are provided in the documentation.

## Background Information

Please refer to our [git wiki](https://github.com/YuzhongHuang/DataScience16CTW/wiki)