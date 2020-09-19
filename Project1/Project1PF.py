import numpy as np, scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from FunctionsDef import FrankePlot, MSE, R2,StdPolyOLS,designMatrix,mean_confidence_interval
import scipy.stats

# Create data
n = 100
poly = 5
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
x, y = np.meshgrid(x,y)

# Setting up the Franke function
z = FrankePlot(x, y).ravel()

# Setting up the polynomial design matrix
designMatrix = designMatrix(x,y,poly)

# Performing ordinary least squares
Y_train_pred, Y_test_pred, betas = StdPolyOLS(designMatrix,z)
# Confidence interval
c1,c,c2 = mean_confidence_interval(betas)
print(c1,c,c2)