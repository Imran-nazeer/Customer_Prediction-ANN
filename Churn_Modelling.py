                          #BANK CUSTOMER DETECTION



#Importing the libraries
import numpy as np
import matplotlib.pyplot
import pandas as pd

#Importing the Dataset
ds = pd.read_csv("Churn_Modelling.csv")
x = ds.iloc[:, 3:13].values
y = ds.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

#Splitting the Dataset Into training and test set
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=0)

#Scaling The train and test of x
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

#Importing the deep learning libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the Classifier
classifier = Sequential()

#Input Layer and the First Hidden Layer
classifier.add(Dense(output_dim = 6, kernel_initializer="uniform", activation="relu",input_dim=11))

#Second Hidden Layer
classifier.add(Dense(output_dim = 6, kernel_initializer="uniform", activation="relu"))

#Output Layer
classifier.add(Dense(output_dim= 1, kernel_initializer="uniform", activation="sigmoid"))

#Compiling the Neural Networks
classifier.compile(optimizer = "adam",loss="binary_crossentropy",metrics=["accuracy"])

#Fitting the ANN to the Training Set
classifier.fit(xtrain,ytrain,batch_size=10,epochs=100)



#Predicting the test set results
y_pred = classifier.predict(xtest)
y_pred = (y_pred<0.5)


#Prediciting For Induvidual Account Holder With Data
new_pred = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred <0.5)


#The Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)





