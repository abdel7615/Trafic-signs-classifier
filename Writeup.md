

[//]: # (Image References)
[image1]: ./writeup_figures/vis1.png "Data Set Visualisation"
[image2]: ./writeup_figures/vis2.png "Histogram"
[image3]: ./writeup_figures/newimages.png "Images from the web"
[image4]: ./writeup_figures/softmax.png "Softmax Probabilities"


# **Traffic Sign Recognition** 

## Writeup 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Data Set Summary & Exploration

#### 1. Basic summary of the data.

I used basic numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32x32x3 (RGB)
* The number of unique classes/labels in the data set is 43 unique classes.

#### 2. Exploratory visualization of the dataset.

To visualise the data set, I took randomly 7 images from the training set and showed them. 

![alt text][image1]

One of the images that come to mind is how many times does a specific sign appear in the data set. To anwser that question I used a histogram of the training set and the test set.

![alt text][image2]

We can see that the distribution of signs is similar in both the training and test stes, which means that the data was very well shuffled before splitting. This raises the following question : will the distribution of the data influence the accuracy of the model ? In other words, what if the model is learning to expect some types of signs more than others ? Is this a bad thing to learn after all ? aren't some signs more common than others anyway ? I believe this question is worth looking into, but unfortunately it's not the purpose of this project. 

### Design and Test a Model Architecture

#### 1. Preprocessing 

The preprocessing used in this project is converting the images to grayscale, the main purpose is to reduce the computing time during the training. The one can argue though that the conversion makes the neural network loose one of the main caracteristics of distinction of the traffic signs which is color. As much as I would like to try both "grayscale" and "RGB" images in training the model in order to study the influence of conversion on performance, I will leave it for future improvement of the project.

The images are also normalized before training the model, this has no visual effect on the images but it is a basic requirement for a good performance.

An other preprocessing technique that could have been used is data augmentation, this has proven to improve the performance significantly as I saw in other collègues projects but since omitting it keeps the results above the requiremnets I will not use it. The main reason for that is to reduce the computing time (I'm using an Intel i3 processor which is very weak for this kind of computing !) 

#### 2. Model architecture

My first model consisted of the basic LeNet architecture we saw in the LeNet lab, but since it gave a low performance of 89% accuracy I added the doprout technique on activation layers to improve performance. The final architecture consists of the following:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding,outputs 10x10x16    |                                          
| Max pooling  		    | 2x2 stride,  outputs 5x5x16        			|
| Flatten				| 5x5x16 to output 400                          |
| Fully Connected       | 400  to output 120                            |
| RELU					|												|
| Dropout               | Keep probability 70%                          |
| Fully Connected       | 120  to output 84                             |
| RELU					|												|
| Dropout               | Keep probability 70%                          |
| Fully Connected       | 120  to output 84                             |



#### 3. Parameters choice

To train the model I used the following parameters :

|  Parameter            | Value            |
|:---------------------:|:----------------:|
|  Optimizer            | Adam optimizer   |
|  Batch size           | 100              |
|  Epochs               | 30               |
|  Learning rate        | 0.001            |
|  Dropout probability  | 30%              |

I chose to keep the Adam optimiser since it gave great results and speed during the LeNet lab for images classification. 

#### 4. The approach taken for finding a solution and getting the validation set accuracy to be at least 0.93

My final model results were:
* training set accuracy of 99.8% ( I was expecting this to be 100%, surprisingly it's not!)
* validation set accuracy of 97% 
* test set accuracy of 93.79%

The first architecture I applied was the LeNet architecture from the LeNet lab, unfortunately after parameter tuning (learning rate, epochs and batch size) the accuracy didn't go above 91% for the validation set.

The next thing I tried was to modify the preprocessing phase by converting the images to YUV color space and using the histogram equalize instead of converting to grayscale. This resulted in very low performance (too much computing power consuming) and lower accuracy. So I decided to go back to grayscale and work on the architechture instead.


At this point, I was trying to figure out a way to improve accuracy without making the architecture more complex. By using the dropout technique after two activation layers the accuracy already imporved significantly and reached 93% even though the starting accuracy was worse than before.

All the work after that was a matter of trial and error to achieve 97% validation accuracy, I tuned the parameters one by one to study the effects of each one of them.

Data set augmentation should give a better validation accuracy but I think it would create overfitting. As I saw from some collegues results who used data augmentation, the validation accuracy was above 99% but the test accuracy was les than 94%. I believe working on the architecture itself instead of augmenting the data should give better results. Comparing the two mentioned approaches should be a great follow up for this project.


### Testing The Model on New Images

#### 1. The shosen images from the web

Here are seven German traffic signs that I found on the web:

![alt text][image3]

I did not expect the model to have any difficulties in identifying these images because they are very clear and without background disturbance, but the results were surprising ...

#### 2. Model's predictions and accuracy

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing      		| No passing   									| 
| Pedestrians 			| General caution								|
| Stop sign				| Speed limit (80km/h)							|
| Slippery road    		| Slippery road					 				|
| Speed limit (60km/h)	| Speed limit (50km/h) 							|
| Traffic signals	    | General caution      							|
| Wild animals crossing | Wild animals crossing		  	        		|




The model was able to correctly guess 3 of the 7 traffic signs, which gives an accuracy of 43%. This accuracy is far from the accuracy of the test set. We can see that the model detects very well the shape of the sign but the content is what it can't efficiently detect for some very close signs. 

#### 3. The top 5 softmax probabilities

The top softmax probabilities can be seen in the following image :

![alt text][image4]

For most images the model was 100% sure of his answer even though two of them were wrong. Surprisingle, the model confused the stop sign with a speed limit sign, it was 90% sure of its "wrong" answer but 10% probability was on the correct sign. A detailed look in the weights and logits would give a very good insight of how the model is perceiving these signs. I will add this point to further improvement of the preject if I find time to come back to it before the end of the term.




```python

```
