import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as skl

def MSE(y_data, y_pred):
    n = np.size(y_data)
    return (np.sum((y_data-y_pred)**2))/n

# Create data
np.random.seed()
ndata = 1000
x = np.linspace(-3,3,ndata).reshape(-1,1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2) + np.random.normal(0,0.1,x.shape)

# Design matrix
poly = 5
X = np.zeros((len(x),poly))
for i in range(0,poly):
    X[:,i] = x**i

# Data split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,train_size=0.75)
standardize = StandardScaler()
standardize.fit(X_train)
standardize.transform(X_train)

