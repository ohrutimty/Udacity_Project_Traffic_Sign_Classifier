#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset_visualization.png "Dataset Visualization"
[image2]: ./examples/equalization.png "Equalization"
[image3]: ./examples/augment_data.png "Augment"
[image4]: ./examples/predict1.png "Predict1"
[image5]: ./examples/predict2.png "Predict2"
[image6]: ./examples/conv1.png "Conv1"

[image11]: ./new_test_image/01.jpg "Traffic Sign 1"
[image12]: ./new_test_image/02.jpg "Traffic Sign 2"
[image13]: ./new_test_image/03.jpg "Traffic Sign 3"
[image14]: ./new_test_image/04.jpg "Traffic Sign 4"
[image15]: ./new_test_image/05.jpg "Traffic Sign 5"
[image16]: ./new_test_image/06.jpg "Traffic Sign 6"
[image17]: ./new_test_image/07.jpg "Traffic Sign 7"
[image18]: ./new_test_image/08.jpg "Traffic Sign 8"
[image19]: ./new_test_image/09.jpg "Traffic Sign 9"
[image20]: ./new_test_image/10.jpg "Traffic Sign 10"
[image21]: ./new_test_image/11.jpg "Traffic Sign 11"
[image22]: ./new_test_image/12.jpg "Traffic Sign 12"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
  - 34799 images
* The size of the validation set is ?
  - 4410 images
* The size of test set is ?
  - 12630 images
* The shape of a traffic sign image is ?
  - (32, 32, 3)
* The number of unique classes/labels in the data set is ?
  - There are 43 classes

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to do histogram equalization to achieve better performance.

Here is an example of a traffic sign image before and after histogram equalization.

![alt text][image2]

Next, normalize all images.

I also generated more data in order to balance each class of datasets.

To add more data to the data set, I used the following techniques including zoom, translation, rotation and affine trasformation.


Here is an example of an original image and an augmented image:

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer    	                |     Description            | Output    |
|:-------------------------:|:--------------------------:|:---------:|
| Input       	            | RGB Image  	               | 32X32X3   |
| Convolution Layer 1 (5X5) | strides 1X1, valid padding | 28X28X16  |
| Relu       	              |   			                   |           |
| Max Pooling       	      | 2X2  			                 | 14X14X16  |
| Convolution Layer 2 (5X5) | strides 1X1, valid padding | 10X10X32  |
| Relu       	              |   			                   |           |
| Max Pooling       	      | 2X2   			               | 5X5X32    |
| Flatten      	            |   			                   |           |
| Fully-connected Layer 1   |   			                   | 360       |
| Relu       	              |   			                   |           |
| Dropout       	          | 0.6 keep probability       |           |
| Fully-connected Layer 2   |   			                   | 120       |
| Relu       	              |   			                   |           |
| Dropout      	            | 0.6 keep probability       |		       |
| Fully-connected Layer 3   |   			                   | 43        |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used LeNet to train. Therefore, most of parameters are not changed. I have modified the batch size to 64 and 256, tried learning rate for several values; however, none of one is able to increase accuracy dramatically.

This is learning rate test result:

| Learning Rate    	|     Accuracy	    |
|:-----------------:|:-----------------:|
| 0.01       	    | 0.059 			|
| 0.005    		    | 0.82 				|
| 0.001				| 0.95				|
| 0.0005      		| 0.94				|
| 0.0001			| 0.91      		|

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy is 0.99.
* validation set accuracy is 0.97.
* test set accuracy is 0.95.

I use LeNet as my model but modified 2 parts.
1. Modified the depth of convolution layer.
    - covolution layer 1: 6 -> 16
    - convolution layer2: 16 -> 32

    When modified the depth, remember that the fully-connected layer output also need to be changed.

    After this step, the accuracy will be raised from 0.91 to 0.93.
2. Add dropout function in fully-connected layer.

    I tried to add dropout function in convolution layer, but the accuracy is down to about 0.75; then I put dropout function in fully-connected layer. The accuracy will increase to 0.95. Moreover, the probability of dropout 0.6 will have the best accuracy, but 0.5 and 0.7 also got closely result.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have got 12 images from the internet. The first 6 images have better resolution, and the rest of the images are taken screenshot from driving recorder videos about German road on YouTube.

![alt text][image11] ![alt text][image12] ![alt text][image13]
![alt text][image14] ![alt text][image15] ![alt text][image16]
![alt text][image17] ![alt text][image18] ![alt text][image19]
![alt text][image20] ![alt text][image21] ![alt text][image22]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| Right-of-way at the next intersection|Right-of-way at the next intersection |
| Go straight or left					| Go straight or left											|
| Speed limit (30km/h)	      		| Speed limit (30km/h)					 				|
| Slippery road			| Slippery Road      							|
| Priority road | Priority road |
| Bumpy road | Bumpy road |
| Keep right | Keep right |
| Speed limit (100km/h) | Speed limit (100km/h) |
| Yield | Yield |
| Turn right ahead | Turn right ahead|
| Vehicles over 3.5 metric tons prohibited | Vehicles over 3.5 metric tons prohibited |


The model was able to correctly guess 12 of the 12 traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image4] ![alt text][image5]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

This is a convolution layer1 visualization for a "turn left ahead" image.
![alt text][image6]
