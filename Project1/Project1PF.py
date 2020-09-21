import numpy as np, scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from FunctionsDef import FrankePlot, MSE, R2,StdPolyOLS,designMatrixFunc,betaConfidenceInterval,Scale
import scipy.stats

#Exercise 1a)

# Create data
n = 1000
poly = 20
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
x, y = np.meshgrid(x,y)

# Setting up the Franke function
z = FrankePlot(x, y, plot = False).ravel() + 0.5*np.random.randn(n**2)

# Setting up the polynomial design matrix
designMatrix, params = designMatrixFunc(x, y, poly)

# Performing ordinary least squares
Y_train_pred, Y_test_pred, Y_train, Y_test, betas,z_pred = StdPolyOLS(designMatrix,z,scalee = False)

# Calculate and print the train and test MSE
print("Training MSE for ordinary least squares: ", MSE(Y_train, Y_train_pred), ".")
print("Test MSE for ordinary least squares: ", MSE(Y_test, Y_test_pred), ".")

# Calculate and print train and test R2
print("Training R2 for ordinary least squares: ", R2(Y_train, Y_train_pred), ".")
print("Test R2 for ordinary least squares: ", R2(Y_test, Y_test_pred), ".")

# Confidence interval
beta_confInt = betaConfidenceInterval(z, n, 1, designMatrix, betas, plot = True)

# Exercise 1b)
MSE_train_poly = np.zeros(poly)
MSE_test_poly = np.zeros(poly)
MSE_z = np.zeros(poly)
for i in range(poly):
    designMatrix_poly, params_poly = designMatrixFunc(x, y, i+1)
    Y_train_pred_poly, Y_test_pred_poly, Y_train_poly,Y_test_poly, betas_poly,z_pred_poly = StdPolyOLS(designMatrix_poly, z, scalee=False)
    MSE_train_poly[i] = MSE(Y_train_poly,Y_train_pred_poly)
    MSE_test_poly[i] = MSE(Y_test_poly,Y_test_pred_poly)
    MSE_z[i] = MSE(z, z_pred_poly)

plt.plot(np.arange(0, poly, 1),(MSE_train_poly),label = "Train MSE")
plt.plot(np.arange(0, poly, 1),(MSE_test_poly),label = "Test MSE")
#plt.plot(np.log10(MSE_z),label = "Z MSE")
plt.suptitle('Training and test MSE as a function of polynomial degree (complexity)', fontsize=25)
plt.ylabel('MSE', fontsize=20)
plt.xlabel('Polynomial degree (complexity)', fontsize=20)
plt.legend(loc="upper right")
plt.show()