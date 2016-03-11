# Project Proposal

## Dataset
Cross sectional MRI brain image data from Open Access Series of Imaging Studies (OASIS)

## Team
Wilson Tang and Yuzhong Huang

## Project Description
Healthcare is an ever evolving field and medical imaging is an essential tool for diagnosing and assessing various conditions from neurodegenerative diseases like Alzheimer’s to characterizing overall health based on factors like brain volume. To understand how MRI images and CDR related and explore more about image processing and deep learning, we built a 6-layer convolutional neural network using theano for a MRI based CDR prediction model.

## Learning Goals
1. Dive into algorithms(neural networks round 2!) and create them by hand as much as possible (inside image processing and machine learning) 

2. Create interesting, insightful representations of data.

3. Learn to use supercomputer to compute complex algorithms (we are planning on creating multiple neural networks and would like to request usage).

## MVP

We will create a basic visualization tool that can create a virtual brain MRI image based their clinical dementia ratings.

## Stretch Goals

We will add another tool to assess an image the user inputs and plot their data against our distribution

## Final Deliverable

Tool or data visualization that will predict patient’s clinical dementia rating (CDR) based on input MRI brain image. And potentially compare input MRI to the generalized virtual MRI brain image. The comparison will also include visualization of patient’s CDR, specifically, where is their brain in the distribution of the brains of their peers and which part of the brain is our model paying attention to. Our stretch goal is to make virtual MRI image generator with a toolbar to adjust age, so that the tool can also be used by doctors who want to double check their patient’s brain status. The visualization tool might also be a good education tool for general audience.


## Workflow and Schedule
In general, the work will be done together in conjunction working either in pair programming or on separate functions. Some work will be done in parallel to try out different avenues when meeting is not possible. This workflow reduces the chances of merge conflicts and ensures that the both of us are on similar pages and learning about the algorithms together. 

#### Mid-project check in: 
we hope to have one or two iterations of the algorithms working. 

#### Timeline: 
Week 1- data exploration + basic algorithm
Week 2 - improving the model and working on putting the tool together 
Week 2.5 - write documentation and polish our project working. 

## Assessment
The final deliverable should be assessed on the lessons learned in creating a convolutional neural network as well as the quality of the tool (usefulness, presentation, usability). 

For lessons learned, we will present our learning journey in the form of ipython notebook. And therefore can be assessed based on functionality code and documentation, as well as writing quality. 

Quality of our tool could be assessed based on easiness to use, straightforwardness and functionality. 