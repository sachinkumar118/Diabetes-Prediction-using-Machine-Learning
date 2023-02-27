# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 22:37:53 2023

@author: sachin kumar
"""

#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data collection and analysis

diabetes_dataset=pd.read_csv('D:/diabetes prediction/diabetes.csv')

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

#seperating data and labels

X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']

#data standardisation

scaler=StandardScaler()
scaler.fit(X)
Standardized_data=scaler.transform(X)

X=Standardized_data
Y=diabetes_dataset['Outcome']


# train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)


#training the model

Classifier=svm.SVC(kernel='linear')
#training support vector vector machine classifier
Classifier.fit(X_train,Y_train)

#model evaluation and accuracy score on training data

X_train_prediction=Classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy score of training data:",training_data_accuracy)

#accuracy of test data
X_test_prediction=Classifier.predict(X_test)
training_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy score of test data:",training_data_accuracy)

#making a predictive system
input_data=(8,183,64,0,0,23.3,0.672,32)
#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the array as we are predicitng only for the single instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#standardize the input data

Std_data=scaler.transform(input_data_reshaped)

prediction=Classifier.predict(Std_data)

if(prediction[0]==0):
    print("The patient is non-diabetic")
else:
    print("The patient is suffering form diabetes")