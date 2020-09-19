import numpy as np, scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from FunctionsDef import FrankePlot, MSE, R2,StdPolyOLS,designMatrix,betaConfidenceInterval,Scale
import scipy.stats

# Create data
n = 100
poly = 5
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
x, y = np.meshgrid(x,y)

# Setting up the Franke function
z = FrankePlot(x, y, plot = False).ravel()

# Setting up the polynomial design matrix
designMatrix = designMatrix(x,y,poly)

# Scaling design matrix by subtracting mean and dividing my variance
designMatrix_scale = Scale(designMatrix)

# Performing ordinary least squares
z_pred, Y_train_pred, Y_test_pred, betas = StdPolyOLS(designMatrix_scale,z)
#print(betas)

# Confidence interval
beta_confInt = betaConfidenceInterval(z, z_pred, n, 1, designMatrix_scale, betas)

fig = plt.figure()
plt.errorbar(np.arange(0,len(betas),1),betas,yerr=beta_confInt[:,0])
plt.show()

