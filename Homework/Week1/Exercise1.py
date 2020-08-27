import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


# Read files
dataFile = pd.read_csv("EoS.csv", header = None)
dataFile.columns = ["Density", "Energy"]
dataFile.info()

# Define the design matrix
X = np.zeros((len(dataFile["Density"]),5))
print(X.shape)

X[:,0] = 1
X[:,1] = dataFile["Density"]**(2/3)
X[:,2] = dataFile["Density"]
X[:,3] = dataFile["Density"]**(4/3)
X[:,4] = dataFile["Density"]**(5/3)

print(X)

# Train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, dataFile["Energy"], test_size=0.75)

# Linear operations
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
ytilde = X_train @ beta

# Prints
print("Training R2")
print(R2(y_train,ytilde))
print("Training MSE")
print(MSE(y_train,ytilde))
ypredict = X_test @ beta
print("Test R2")
print(R2(y_test,ypredict))
print("Test MSE")
print(MSE(y_test,ypredict))