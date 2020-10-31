# Project 4: Behavioral Cloning



## **Project Goals: **

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Required Files und Quality of Code: 

#### 1. Are all required files submitted?

The following files are submitted in the Github repository:

- **model.py** containing the script to create and train the model
- **drive.py** for driving the car in autonomous mode
- **model.h5** containing a trained convolution neural network 
- **writeup.md** summarizing the results
- **finalRun.mp4** shows the behavior of the AV on the first test track

#### 2. **Is the code functional?**

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

There haven´t been any issues reported while running the simulator in autonomous mode.  

#### 3. Is the code usable and readable? 

The code in model.py is structured as follows: 

1. Import necessary packages (e.g. numpy, csv, keras, tensorflow)
2. Data Generation: 
   - Load data from drive log
   - Split data into training and validation set
   - Implement a data generator for both training and validation data
3. Build the model archtecture
4. Train the model

This structure is maintained in the comments. 



## Model Architecture and Training Strategy: 

#### 1. **Has an appropriate model architecture been employed for the task?**

The idea was to use a high capacity model in order to prevent underfitting. That´s why the final architecture leans to the one which has been proposed in the class and used by the Nvidia researchers for implementing end to end autonomous driving. It comprises 5 convolutional layers and 3 concatenated fully connected layers. Max pooling has been used in order to reduce the feature map dimensions. Relu serves as activation function. The architecture comprises the following layers: 

- Preprocessing (lambda) layer for normalization**, **mean centering and image cropping
- Convolutional Layer 5x5 @ 24

- Relu Activation
- Max Pooling 2x2
- Convolutional Layer 5x5 @ 36
- Relu Activation
- Max Pooling 2x2
- Convolutional Layer 5x5 @ 48
- Relu Activation
- Max Pooling 2x2
- Convolutional Layer 3x3 @ 64
- Relu Activation
- Max Pooling 2x2
- Convolutional Layer 3x3 @ 64
- Relu Activation
- Max Pooling 2x2
- Flatten Layer 
- Dense Layer
- Relu Activation
- Dense Layer 
- Relu Activation
- Dense Layer (Output)

#### 2. **Has an attempt been made to reduce overfitting of the model?**

The data was splited up into training and validation set by a ratio of 80/20. Both the training and validation errors have been decreasing during the entire training process. That´s why no dropout layers have been added to the model architecture. In order to reduce overfitting data augmentation has been applied by flipping the camera images. Another method to reduce overfitting was cropping the images in order to get rid of the background noise. In general, increasing the training set is also a reasonable approach to mitigate overfitting. This has been achieved by incorporating not only the center image, but also using the left and right ones.  

#### 3. **Have the model parameters been tuned appropriately?**

The model was trained utilizing an adam optimizer, so the learning rate was not tuned manually. The batch size was chosen to be 32. Training was done for 10 epochs. Mean squared error was used as loss function.   

#### 4. **Is the training data chosen appropriately?**

On the given hardware it was rather hard to smoothly run the simulator. That´s why the training data quality was poor. Therefore, the Udacity provided training set was used, which comprises data from the first test track. Like described above, data augmentation techniques have been applied in order to increase the training set. 



## Architecture and Training Documentation: 

#### 1. Are solution design and model architecture documented

The model architecure with its layers is described above. A similiar architecutre has shown promising results in the past for end to end learning of human driving behavior. That´s why the here implemented architecure leans to this approach. The convolutional layers solve the perception task, i.e. they identify features on the input images like edges and even objects in the higher layers. Nonlinearties are incorporated in the model by applying Relu activation functions. Ulitmately, the fully connected (dense) layers transform perception features into a steering information. 

#### 2. Is the creation of the training dataset and training process documented?**

The Udacity provided training data has been used as initial set to apply data augmentation, like described above. The example image below shows a raw input image from the center camera, i.e. no preprocessing has been applied. One raw data image has a dimension of 160x320x3. After applying cropping, it´s reduced to 80x320x3, which is the ultimate input size for the first convolutional layer. 

[]: https://github.com/dschmoeller/04BehavioralCloningDeepNNsKeras/blob/master/examples/rawImageExample.jpg

A data generator is used to provide batches with a size of 32. The training is done for 10 epochs. Mean squared error is utilized as loss function. The training data is randomized. After 10 epochs of training, the validation error went down to 0.086. A simulation of the corresponding model shows promising results.     



## Simulation: 

#### 1. **Is the car able to navigate correctly on test data?**

Below is a link to the autonomous test drive simulation on the first track. The behavior of the AV looks promising and holds all requirements.  

[]: https://github.com/dschmoeller/04BehavioralCloningDeepNNsKeras/blob/master/finalRun.mp4




