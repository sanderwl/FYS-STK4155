import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def MSE(y_data, y_pred):
    n = np.size(y_data)
    return (np.sum((y_data-y_pred)**2))/n

# Create data
ndata = 1000
x = np.random.rand(ndata)
y = 2+5*x+x+0.1*np.random.rand(ndata)

# Design matrix
poly = 3
X = np.zeros((len(x),poly))
for i in range(0,poly):
    X[:,i] = x**i

# Data split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,train_size=0.75)
standardize = StandardScaler()
standardize.fit(X_train)

# Ordinary least squares
LeastSquares = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
print("Beta coefficients: " , LeastSquares)

# Predict
Y_train_pred = X_train @ LeastSquares
Y_test_pred = X_test @ LeastSquares

print("Training MSE: " , MSE(Y_train,Y_train_pred), ".")
print("Test MSE: " , MSE(Y_test,Y_test_pred), ".")
