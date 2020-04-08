#!/usr/bin/env python
# coding: utf-8

# In[20]:


#COLOR_QUANTIZATION - Representing an image only in some k number of components only.
#Done with K-Means clustering. K=2 here i.e there are 2 clusters and 3 colour componenets.

import numpy as np
import cv2

img = cv2.imread(r"C:\Users\shriya-student\Documents\machinelearning\messi.jpg")
Z = img.reshape((-1,3)) #Making it 3D shape

#We're converting to float
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#In-build criteria to setup number of iterations to find means ig.
"""cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached. 
cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter. 
cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met."""
#1.0 means no stop in between(Accuracy)

K = 2
ret, label, center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#Random centers for K means are random as we dont know which colours we have on image

center = np.uint8(center)
#converts image back to integer.
res = center[label.flatten()]
res2 = res.reshape((img.shape))
#print(center)
#print(label)
#print(res)
cv2.imshow("res2", res2)
cv2.waitKey(0)

"""
Centers is array containing the RGB colour values for the K centers. 
Label is the label for each pixel in the image. If it is center[0] or center[k-1] etc.
res forms array of length no. of pixels where each element is center[i] where i was the value of label.
"""

