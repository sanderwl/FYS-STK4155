import numpy as np, scipy.stats as st
from random import shuffle, sample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from FunctionsDef import inputsss, FrankeFunc, MSE, R2,StdPolyOLS,designMatrixFunc, designMatrixFunc2,\
    betaConfidenceInterval,Scale, sciKitSplit, MSEplot, calc_Variance, calc_bias, BVPlot, Scale2, realSplit,FrankPlot
import scipy.stats
from mlxtend.evaluate import bias_variance_decomp

# Input function
observations, degree, scaleInp, figureInp, part, noiseInp, noiseLVL,testsizesss = inputsss()

# Exercise 1a)

# Create data
n = int(observations)
poly = int(degree)
x = np.arange(0, 1, 1 / n)
y = np.arange(0, 1, 1 / n)
xx, yy = np.meshgrid(x, y)

# Setting up the Franke function

z = FrankeFunc(x, y,) + float(noiseLVL) * np.random.randn(n)
FrankPlot(xx, yy, plot=figureInp)


# Setting up the polynomial design matrix
designMatrix = designMatrixFunc2(x,y,poly,noiseLVL)

# Splitting data into training and test sets
testSize = float(testsizesss)
#X_train, X_test, Y_train, Y_test = sciKitSplit(designMatrix, z, testSize, True)

#Scale data
#X_train_scale, X_test_scale = Scale(X_train, X_test, scalee = scaleInp)

if (str(part) == "a" or str(part) == "all"):

    # Performing ordinary least squares
    Y_train_pred, Y_test_pred, betas = StdPolyOLS(X_train_scale, X_test_scale, Y_train, Y_test)

    # Calculate and print the train and test MSE
    print("Training MSE for ordinary least squares: ", MSE(Y_train, Y_train_pred), ".")
    print("Test MSE for ordinary least squares: ", MSE(Y_test, Y_test_pred), ".")

    # Calculate and print train and test R2
    print("Training R2 for ordinary least squares: ", R2(Y_train, Y_train_pred), ".")
    print("Test R2 for ordinary least squares: ", R2(Y_test, Y_test_pred), ".")

    # Confidence interval
    beta_confInt = betaConfidenceInterval(z, n, 1, designMatrix, betas, plot = figureInp)

elif (str(part) == "b" or str(part) == "all"):
    # Exercise 1b)
    MSE_train_poly = np.zeros(poly)
    MSE_test_poly = np.zeros(poly)
    bias = np.zeros(poly)
    variance = np.zeros(poly)
    MSE_z = np.zeros(poly)
    rn = 0
    for i in range(poly):
        designMatrix_p = designMatrixFunc2(x, y, i+1, noiseLVL)
        designMatrix_poly = Scale2(designMatrix_p,scalee=scaleInp)
        X_train_scale_poly, X_test_scale_poly, Y_train_poly, Y_test_poly, randomNumber = realSplit(designMatrix_poly,z,testSize,i,rn,True)
        rn = randomNumber
        Y_train_pred_poly, Y_test_pred_poly, betas_poly = StdPolyOLS(X_train_scale_poly, X_test_scale_poly, Y_train_poly, Y_test_poly)
        bias[i] = calc_bias(Y_test_poly, Y_test_pred_poly, n)
        variance[i] = calc_Variance(Y_test_poly, Y_test_pred_poly, n)
        MSE_train_poly[i] = MSE(Y_train_poly, Y_train_pred_poly)
        MSE_test_poly[i] = MSE(Y_test_poly, Y_test_pred_poly)

    if figureInp == True:
        MSEplot(MSE_train_poly, MSE_test_poly, poly)
        BVPlot(bias, variance, MSE_test_poly, poly)

    print(MSE_test_poly - bias**2 - variance)

elif (str(part) == "c" or str(part) == "all"):
    cvin = input("Number of folds in CV? (integer\False)")

    # Using CV to create test and train data
    Y_train_pred_cv, Y_test_pred_cv, Y_train_cv, Y_test_cv, betas_cv, z_pred_cv = StdPolyOLS(designMatrix, z, scalee=scaleInp, cv=int(cvin))

    if figureInp == True:
        plt.plot(np.arange(0, poly, 1),(MSE_train_poly),label = "Train MSE")
        plt.plot(np.arange(0, poly, 1),(MSE_test_poly),label = "Test MSE")
        #plt.plot(np.log10(MSE_z),label = "Z MSE")
        plt.suptitle('Training and test MSE as a function of polynomial degree (complexity)', fontsize=25)
        plt.ylabel('MSE', fontsize=20)
        plt.xlabel('Polynomial degree (complexity)', fontsize=20)
        plt.legend(loc="upper right")
        plt.show()



