
# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: Figure_1.png "train"
[image2]: ./data/IMG/center_2016_12_01_13_30_48_287.jpg "centre"
[image3]: /data/IMG/left_2016_12_01_13_30_48_287.jpg "centre"
[image4]: /data/IMG/right_2016_12_01_13_30_48_287.jpg "centre"


 
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 demonstraiting the performance of the CNN
* Readme.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

Initially all the image data is normalized by a lamda layer (x / 255.0 - 0.5), then the image is cropped so only the most relevant information remains to be processed. I cropped the image 60 pixels on top and 20 on the bottom so the road can be processed.
My model consists of four convolutional layer and three fully connected layers. 
First Convolutional layer
* 5x5 kernel, 2x2 stride, 24 output channels, relu activation
Second Convolutional layer
* 5x5 kernel, 2x2 stride, 36 output channels, relu activation
Third Convolutional layer
* 5x5 kernel, 2x2 stride, 48 output channels, relu activation
Fourth Convolutional layer
* 3x3 kernel, 1x1 stride, 64 output channels, relu activation
Fifth Convolutional layer
* 3x3 kernel, 1x1 stride, 64 output channels
Flatten
First Fully conected layer
* Output 100
Second Fully conected layer
* Output 50
Output layer
* Output 1



#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 82 and 88).  This dropout are located between the second and third and fourth and fifth convolutional layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 16-17). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the provided data and augment it through different techniques. 

My first step was to use a convolution neural network model similar to the suggested by Nvidia I thought this model might be appropriate because it's use for the same purpose. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I modified the model by adding drop outs as shown before. Here a figure of the training:

![alt text][image1]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track at the begining to improve the driving behavior in these cases, I modified the bias induced to the steering of the side cameras.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I used the provided information, I use the center and the side cameras images. Here is an example images:

![alt text][image2]
![alt text][image3]
![alt text][image4]

For the left image the steering angle was set to the original value plus 0.218 while for the right image the same value was substracted from the original steering angle. 
To augment the data set, I also flipped images and angles thinking that this would balance the training information since in trajectory there were more right turns

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4, more epoch caused the overfitting of the model. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 3. Improvements
I collected a lot of data but I realized that the test was done by keeping the same speed of the vehicle. This made my data not useful since I used to reduce the speed on the curves. 

