#!/usr/bin/env python
# coding: utf-8

# In[5]:

import numpy as np
import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(r'C:\Users\shriya-student\Documents\machinelearning\frontal_face.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\shriya-student\Documents\machinelearning\eye_face.xml')
#Contains entire model of face and eye(.xml).
#Weights of probablity nodes are saved on .xml file.

img = cv2.imread(r'C:\Users\shriya-student\Documents\machinelearning\us.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)  #detects faces
#Detects img parts and stores finest imgs whose weights match .xml file.

for(x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y), (x+w,y+h), (255,0,0), 2)
    #Draws rectangle around the face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray) #detects eyes
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0),2)
        #Draws rectangle around the eyes.
    
cv2.imshow("img",img)
cv2.waitKey(0)

