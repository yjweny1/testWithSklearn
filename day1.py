# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sklearn as skl
#from sklearn.preprocessing import Imputer
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [0])
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import StandardScaler

#import data from local
dataset = pd.read_csv('Data.csv')
dataset.head()
x = dataset.iloc[ : , :-1].values
y = dataset.iloc[ : , 3].values

#clean data handle missing value in x 
imputer = skl.Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(x[ : , 1:3])
x[ : , 1:3] = imputer.transform(x[ : , 1:3])

# encode 
labelencoder_x = skl.LabelEncoder()
x[ : , 0] = labelencoder_x.fit_transform(x[ : , 0])

#create fake data
onehotencoder = skl.OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = skl.LabelEncoder()
y =  labelencoder_y.fit_transform(y)

#split data and 80 use to train and 20 to predict
x_train, x_test, y_train, y_test = skl.train_test_split( x , y , test_size = 0.2, random_state = 0)

#test
sc_x = skl.StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)