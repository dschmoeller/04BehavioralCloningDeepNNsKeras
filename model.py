import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import csv as csv
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout 
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split


### D A T A   G E N E R A T I O N   
### 1.) Load the data from the driving log file
# The csv file is structured like this: 
# center - left - right - steering - throttle - brake - speed
logFileLines = []
with open ("./data/driving_log.csv") as log: 
    reader = csv.reader(log)
    next(reader)
    for line in reader: 
        logFileLines.append(line)   
### 2.) Split the data into training and validation set
trainingData, validationData = train_test_split(logFileLines, test_size=0.2)        
### 3.) Define a generator which provides data batches more (memory) efficiently than just loading and storing the entire data set
def dataGenerator(data, batchSize=32): 
    numDataSamples = len(data)
    while True:
        # Randomize data
        np.random.shuffle(data)
        # Return (i.e. yield) a batch every time the dataGenerator gets called
        for offset in range(0, numDataSamples, batchSize):
            batchData = data[offset:offset+batchSize]
            # Extract image links for center, left and right images
            # Extract steering values 
            centerImgLinks = []
            leftImgLinks = []
            rightImgLinks = []
            steeringCenter = []
            steeringLeft = []
            steeringRight = []    
            for line in batchData: 
                centerImgLinks.append("./data/" + line[0])
                leftImgLinks.append("./data/" + (line[1])[1:])
                rightImgLinks.append("./data/" + (line[2])[1:])
                # Use left and right camera images to pretend the AV is swerved to either left or right
                # Adapt the steering by a correction factor of 0.2 in order to get the AV back to the center
                steeringCenterValue = float(line[3])
                steeringLeftValue = steeringCenterValue + 0.2
                steeringRightValue = steeringCenterValue - 0.2
                steeringCenter.append(steeringCenterValue)
                steeringLeft.append(steeringLeftValue)
                steeringRight.append(steeringRightValue)
                # Load actual images
                centerImages = []
                leftImages = []
                rightImages = []
                for centerImgLink, leftImgLink, rightImgLink in zip(centerImgLinks, leftImgLinks, rightImgLinks): 
                    centerImages.append(plt.imread(centerImgLink))
                    leftImages.append(plt.imread(leftImgLink))
                    rightImages.append(plt.imread(rightImgLink))
                # Stack images and steering values together respectively
                images = centerImages + leftImages + rightImages
                steerings = steeringCenter + steeringLeft + steeringRight
                # Augment the data by flipping the image and inverse the corresponding steering  
                augmentedImages = []
                augmentedSteerings = []
                for img, steerVal in zip(images, steerings): 
                    flippedImg = np.fliplr(img)
                    flippedSteerVal = - steerVal
                    augmentedImages.append(img)
                    augmentedImages.append(flippedImg)
                    augmentedSteerings.append(steerVal)
                    augmentedSteerings.append(flippedSteerVal)             
            # Return (yield) the training batch    
            X_train = np.array(augmentedImages) 
            y_train = np.array(augmentedSteerings)
            yield sklearn.utils.shuffle(X_train, y_train) 

            
### B U I L D   T H E   M O D E L   A R C H I T E C T U R E  
model = Sequential()
# L a y e r   0   (P R E P R O C E S S I N G) 
# Lambda layer as preprocessing unit (normalization and mean centering)
# Cropping layer to remove the above part of the images (which might be rather noise for the NN) 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20), (0,0))))
# L a y e r   1
# Convolution and MaxPool --> Input: 80x320x3 --> Layer 1 --> Output: 40x160x24 
model.add(Conv2D(kernel_size=(5,5), filters=24, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   2
# Convolution and MaxPool --> Input: 40x160x24 --> Layer 2 --> Output: 20x80x36 
model.add(Conv2D(kernel_size=(5,5), filters=36, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   3
# Convolution and MaxPool --> Input: 20x80x36 --> Layer 3 --> Output: 10x40x48
model.add(Conv2D(kernel_size=(5,5), filters=48, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   4
# Convolution and MaxPool --> Input: 10x40x48 --> Layer 4 --> Output: 5x20x64
model.add(Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   5
# Convolution and MaxPool --> Input: 5x20x64 --> Layer 5 --> Output: 2x10x64
model.add(Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   6
# Flatten Layer --> Input: 2x10x64 --> Layer 4 Output: 1280
model.add(Flatten())
# L a y e r   7
# Dense (Fully Connected) and Relu --> Input 1280 --> Layer 7 --> Output: 320
model.add(Dense(320))
model.add(Activation('relu'))
# L a y e r   8
# Dense (Fully Connected) and Relu --> Input 320 --> Layer 8 --> Output: 160
model.add(Dense(160))
model.add(Activation('relu'))
# L a y e r   9   
# Dense (Fully Connected)  --> Input 160 --> Layer 9 --> Output: 16
model.add(Dense(16))
model.add(Activation('relu'))
# L a y e r   10   (O u t p u t)
# Dense (Fully Connected)  --> Input 16 --> Layer 10 --> Output: 1
model.add(Dense(1))


### T R A I N   T H E   M O D E L
# Define data generator for training and validation batches
batchSize = 32
trainingDataGenerator = dataGenerator(trainingData, batchSize)
validationDataGenerator = dataGenerator(validationData, batchSize)
# Use mean squared error function as loss and the adam optimizer (stochastic gradient descent)
model.compile(loss="mse", optimizer="adam")
# Training
behavioralCloningModel = model.fit_generator(trainingDataGenerator, steps_per_epoch=np.ceil(len(trainingData)/batchSize), \
    validation_data=validationDataGenerator, validation_steps=np.ceil(len(validationData)/batchSize), \
    epochs=10, verbose=1)
# Save the model
model.save("model.h5")



