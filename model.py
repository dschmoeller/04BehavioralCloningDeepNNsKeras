import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import csv as csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout 
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


### L O A D   T H E   D A T A 
# Load the data from the driving log file
# The csv file is structured like this: 
# center - left - right - steering - throttle - brake - speed
logFileLines = []
with open ("./data/driving_log.csv") as log: 
    reader = csv.reader(log)
    next(reader)
    for line in reader: 
        logFileLines.append(line)   
# Extract the center image links from the log file
# Extract the steering from the log file
centerImgLinks = []
steering = []
for line in logFileLines: 
    centerImgLinks.append("./data/" + line[0])
    steering.append(float(line[3]))
# Load the actual image using pyplot (in RGB format)
centerImages = []
for link in centerImgLinks: 
    centerImages.append(plt.imread(link))
# Convert the images and the steering angles into numpy format (due to Keras) and build the training set
# Image shape: 160x320x3
X_train = np.array(centerImages)
y_train = np.array(steering)


### B U I L D   T H E   M O D E L 
model = Sequential()
# L a y e r   0
# Lambda layer as preprocessing unit (normalization and mean centering)
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# L a y e r   1
# Convolution and MaxPool --> Input: 160x320x3 --> Layer 1 --> Output: 80x160x24 
model.add(Conv2D(kernel_size=(5,5), filters=24, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   2
# Convolution and MaxPool --> Input: 80x160x24 --> Layer 2 --> Output: 40x80x36 
model.add(Conv2D(kernel_size=(5,5), filters=36, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   3
# Convolution and MaxPool --> Input: 40x80x36 --> Layer 3 --> Output: 20x40x48
model.add(Conv2D(kernel_size=(5,5), filters=48, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   4
# Convolution and MaxPool --> Input: 20x40x48 --> Layer 4 --> Output: 10x20x64
model.add(Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   5
# Convolution and MaxPool --> Input: 10x20x64 --> Layer 5 --> Output: 5x10x64
model.add(Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
# L a y e r   6
# Flatten Layer --> Input: 5x10x64 --> Layer 4 Output: 3200
model.add(Flatten())
# L a y e r   7
# Dense (Fully Connected) and Relu --> Input 3200 --> Layer 7 --> Output: 320
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
# Use mean squared error function as loss and the adam optimizer (stochastic gradient descent)
model.compile(loss="mse", optimizer="adam")
# Train on the (randomized) training set for 10 epochs and use 20% as validation set 
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
# Save the model
model.save("model.h5")



