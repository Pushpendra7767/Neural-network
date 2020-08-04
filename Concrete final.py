# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:42:46 2020

@author: ACER
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda
from keras.utils import plot_model

Concrete = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Neural network\\concrete.csv")
Concrete.head()

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

column_names = list(Concrete.columns)
predictors = column_names[0:8]
target = column_names[8]

first_model = prep_model([8,50,1])
first_model.fit(np.array(Concrete[predictors]),np.array(Concrete[target]),epochs=9)
pred_train = first_model.predict(np.array(Concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,Concrete[target],"bo")
np.corrcoef(pred_train,Concrete[target])

from ann_visualizer.visualize import ann_viz

ann_viz(first_model, title="concrete neural network")









































































