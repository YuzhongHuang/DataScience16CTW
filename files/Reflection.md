# Reflection

####Assessment evidence and interpretation: 
	When you turned in your project proposal you detailed an assessment plan for your project.  In your final reflection, please provide any evidence that you think will be helpful in evaluating your project.   If the only relevant piece of evidence is your final output, that is totally fine, however, if there are other pieces of evidence (visualization mockups, ipython notebooks from models you didn't wind up using, data explorations, etc.) make sure to point those out.  In addition to listing these sources of evidence, please provide a brief interpretation of how they help inform the assessment of your project.

To assess the work please checkout the git pages in our repository with a focus on the impact and conclusions we drew using the saliency map, trying out our predictive model tool, and looking at the convolutional neural network model we built! 

The first two give a good indication of whether our project managed to relate to real-world changing concepts by providing the meaningful data visualization and predictive tool. The third assesses our goal to dive into neural networks and complex algorithms. We focused on predicting one specific indication clinical dementia rating b/c it is the most impactful parameter to predict.


#### Changing the world: 

	Do you think your project has the potential to change the world?  If not, why?  If so, what are the next steps to make this happen?

At this point our project has potential similar to research to add to the knowledge of the world and offer methods for improving our understanding of the possibility of the application of machine learning predictive models (specifically convolutional neural network) . We believe that work in this space is important because diagnosing dementia is something that is currently underutilized (45% of people aware of their condition) and despite the current methodology and human being fairly accurate (~80%). We think in the future when imaging is much more common, a semi-automated system could assist busy clinicians with diagnosing this important disease. In addition it is possible that the saliency map of the convolutional neural network can point to potential indicators and relationships between visual cues offered by the brain and dementia previously not explored or identified by human clinicians. 


####Learning goals: 

	Did you learn the things that you wanted in this project?  If not, why?  If so, why do you think you were successful?

In this project we managed to successfully dive into convolutional neural networks: learning about the concepts of convolution, max-pooling, and feed-forward networks. 

From building a convolutional neural network, we have a deeper understanding of how neural network works. Our implementation is not always correct. Therefore, we learned a lot from our mistakes. 

We once had a bug that our model predicts exactly the same (may be a little bit difference in probabilities) prediction for every data entries and therefore got the same score every time. In the end, we figured out that our learning step is too big for our network. We originally thought that if the learning step is too large, the cost will grow. However, the cost is still reducing but the model is overgeneralizing. That’s very different from having a large learning step in other models.

Secondly, we tried a bunch of interesting techniques in our model. For example, we used regularization term in our cost function to reduce the effect of overfitting. We also used cross entropy validation as our cost function instead of sigmoid activation function to speed up the learning of deep networks.  

Apart from learning we did from convolutional neural network, we also learned how to create a saliency map to visualize the import features in an image. The idea is straightforward while it takes some time for the implementation though. Saliency map is a very interesting and useful tool to value and visualize our model.

Our tool is very straightforward and easy to implement since we already had the most import backend module done. The tool takes the input image and returns prediction from the model. The tool will also shows the saliency map built from the image.

In addition, after building and training our neural network we learned a lot when thinking of how to best link the model we created to relevant world issues (in our case: dementia). Our goal was to create a website that could act as both an informative site with include visualizations that could inform the public but also provide insights into the relationships 
between areas of brain and dementia identified by our model utilizing a saliency map. This information can be used to both improve the model by referring to clinical literature but also potentially suggest trends in dementia patients previously unnoticed.

We think our learning is successful because of our rigorous schedule, work coordination, and usage of online resources. We set a very ambitious schedule to learn and make a convolutional neural network model by referring to multiple online resources and libraries such as Theano within a week. The next week we works on iterating on the model, making a saliency map, making the data visualization tool, and pulling together online resources on dementia to create an informative website on our project. By utilizing this rigorous schedule we were able to get everything done on time and not only learn a lot about multiple aspects of convolutional neural networks, bash, and assessing convolutional models but also learn a lot about clinical diagnosis of dementia and magnetic resonance imaging. In addition we were able to both pair program and work in parallel + explain the work and lessons learned in order to get everything done and make sure we both learned. Finally, we were flexible in the usage of online resources to make sure that we didn’t get too bogged down in the resources and were able to adjust the depth of our learning based on availability. 
