#!/usr/bin/env python
# coding: utf-8
# In[16]:

import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist  #keras has some inbuilt common datasets..mnist is one such dataset.
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
#diff imports from keras

# In[26]:

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
cmp = plt.get_cmap('gray')
plt.show()

#xtrain means the imgs and actual pics.
num_pixels = X_train.shape[1]*X_train.shape[2]  #height*width

X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
#flatting the images which are 28x28 into a 1-Dimensional vector of length 28x28=784 and type float


#normalize your inputs from 0-255 to 0-1(bringing all to uniform span of values)
X_train = X_train/255
X_test = X_test/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#Converts the data into a binary matrix.
#if the matrix[i][j] = 1, it means that the ith picture falls into category of j

num_classes = y_test.shape[1] #no. of classes in y

# In[27]:
#creaating our model
def digit_model():
    model = Sequential() #empty model
    model.add(Dense(num_pixels,input_dim=num_pixels,kernel_initializer="normal", activation="relu"))
    #dense layer added. input dimension=num_pixels.kernet intializer if u want kernel change or anything, default is normal
    #activation function is relu(persons name and type of activation function)
    #Relu(x) = max(0,x) (removes negative values and all)
    
    model.add(Dense(num_classes, kernel_initializer="normal", activation="softmax"))
    #Training on lots of classes and bringing it straight down to num_classes isn't fair
    #Eg:Going from 10,000 classes to 5 classes is not good.
    #The above dense layer addition ensures a more gradual reduction in no. of classes.
    #Eg: From 10,000 to 5,000, to 2,000 and so on :)
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #actually binds and brings model together. 
    #You will lose some features when passing through each layer.
    #You can define how loss is measures(categorical_crossentropy). It is used for single label categorization. 
    #Some optimizers like Adam etc are there to help,they aren't needed as such. 
    #You can change them and check the variance in accuracy(minute variance).
    #metrics: on what grounds do u want to evaluate model. Our case its accuracy(based on elements).
    #Other options could include categorical_accuracy(based on classes), top k categorical accuracy(usually for complex qns)
    return model


#_main_
#Build the model

model = digit_model()

#Fit/rain the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
#epoch is each instance the model is passed over training set. More epochs are better mostly. 
#If bad training data and according to problem, then too many epochs can cause overfitting
#batch_size is how much data u want to load at one time. Especially if large amt of training data,
#taking all the data at once is not viable. So the batch_size decides no. of samples loaded at once.
#increase in batch_size can decrease accuracy

#Final evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)
print("Baseline Error: %2f%%"%(100-scores[1]*100))


# In[ ]:

img = cv2.imread(r'C:\Users\shriya-student\Documents\machinelearning\six.png',0)
img = img.reshape(img.shape[0],num_pixels).astype('float32')
img = img/255
pred = model.predict([img])
print(pred)



