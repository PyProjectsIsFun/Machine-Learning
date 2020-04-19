#!/usr/bin/env python
# coding: utf-8

# In[76]:


#Importing the usual modules of numpy, matplotlib, pandas etc.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#Used for splitting data into training and testing
from sklearn.neighbors import KNeighborsClassifier
#KNN Model
from sklearn.tree import DecisionTreeClassifier
#Decision Tree Model
diabetes = pd.read_csv(r"C:\Users\shriya-student\Documents\machinelearning\diabetes.csv")
#diabetes.head()

#diabetes.info() #gives entries data types memory etc.


# In[77]:


#Classifier using KNN
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns!="Outcome"],diabetes["Outcome"],stratify=diabetes["Outcome"], random_state=66)
#.loc opens the columns of the data frame diabetes. It is assigning all rows except "Outcome" to the X_train and X_test
#while it assigns the outcomes column for y_train and y_test.
#Random state in basic words is a value for how "randomized" it will be in splitting the data into training and testing.
#random state = 66 also implies that 2/3 of the data is for training and 1/3 for testing which is an appropriate combination, but this is not the case always.
#stratify means that the data is arranged using "outcomes" column as labels.


training_accuracy=[]
test_accuracy=[]
#finding best k
neighbors_settings = range(1,200)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    #trains
    training_accuracy.append(knn.score(X_train,y_train))
    test_accuracy.append(knn.score(X_test,y_test))
    #appending accuracy of the model on both training and testing data.

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="testing accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#Best is k=9
knn = KNeighborsClassifier(n_neighbors=9)
#(9,10) or (18,19) seems most appropriate.
knn.fit(X_train,y_train)
#Training model
print("Training Accuracy: {:.2f}".format(knn.score(X_train,y_train)))
print("Testing Accuracy: {:.2f}".format(knn.score(X_test,y_test)))
#Printing training and testing accuracy.

# In[75]:

tree = DecisionTreeClassifier(random_state=0)
#creating model. You can also initialize values such as max_depth etc.
tree.fit(X_train,y_train)
#training the model
print("Training Accuracy: "+ " " + str(tree.score(X_train,y_train)))
print("Testing Accuracy: "+ " " + str(tree.score(X_test,y_test)))

plt.figure(figsize=(8,6))
#making graph of size 8x6 on y axis- x axis.
n_features = 8 #no. of features
plt.barh(range(n_features), tree.feature_importances_,align='center')
#plotting importances of all features
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)
plt.show()
#showing graph.


