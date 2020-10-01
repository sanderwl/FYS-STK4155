import numpy as np, scipy.stats as st
from random import shuffle, sample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from FunctionsDef import inputsss, FrankeFunc, MSE, R2,StdPolyOLS,designMatrixFunc, designMatrixFunc2,\
    betaConfidenceInterval,Scale, sciKitSplit, MSEplot, calc_Variance, calc_bias, BVPlot, Scale2, realSplit,FrankPlot, bootStrap, CV, MSEplotCV, realSplit2, FrankPlotDiff
import scipy.stats
from mlxtend.evaluate import bias_variance_decomp
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

# Input function
observations, degree, scaleInp, figureInp, part, noiseInp, noiseLVL,testsizesss = inputsss()

# Exercise 1a)

# Create data
n = int(observations)
poly = int(degree)
x = np.arange(0, 1, 1 / n)
y = np.arange(0, 1, 1 / n)
xx, yy = np.meshgrid(x, y)

# Test size throughout script
testSize = float(testsizesss)

# Setting up the Franke function with/without noise
z = FrankeFunc(xx, yy) #+ float(noiseLVL) * np.random.randn(n)

if (str(part) == "a" or str(part) == "all"):

    # Setting up the polynomial design matrix with/without noise
    designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)

    # Scaling the data by subtracting mean and divide by standard deviation
    designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

    # Splitting the data into training and test sets
    rn = 0
    X_train, X_test, Y_train, Y_test, randomNumber = realSplit2(designMatrix_scale, z, testSize, 0, rn, True)

    # Performing ordinary least squares (OLS)
    Y_train_pred, Y_test_pred, betas = StdPolyOLS(X_train, X_test, Y_train, Y_test)

    # Calculate and print the train and test MSE
    print("Training MSE for ordinary least squares: ", MSE(Y_train, Y_train_pred), ".")
    print("Test MSE for ordinary least squares: ", MSE(Y_test, Y_test_pred), ".")

    # Calculate and print train and test R2
    print("Training R2 for ordinary least squares: ", R2(Y_train, Y_train_pred), ".")
    print("Test R2 for ordinary least squares: ", R2(Y_test, Y_test_pred), ".")

    # Confidence interval
    beta_confInt = betaConfidenceInterval(z, n, 1, designMatrix_scale, betas, plot = figureInp)

    # Plotting Franke function
    FrankPlot(xx, yy, z, plot=figureInp)

    #Plotting the estimated Franke function
    estimated = designMatrix_scale @ betas
    FrankPlot(xx, yy, estimated, plot=figureInp)

    #Plotting difference between real and predicted Franke function
    FrankPlotDiff(xx,yy,z,estimated, plot = figureInp)


elif (str(part) == "b" or str(part) == "all"):
    # Exercise 1b)
    B = input("This exercise utilizes a bootstrap resampling method that generates B sets of data of size n. How many sets should be generated? (integer)")
    bias = np.zeros(int(B))
    variance = np.zeros(int(B))
    MSE_train_poly = np.zeros(int(B))
    MSE_test_poly = np.zeros(int(B))
    biasBoot = np.zeros(poly)
    varianceBoot = np.zeros(poly)
    MSE_train_boot = np.zeros(poly)
    MSE_test_boot = np.zeros(poly)
    rn = 0
    for i in range(poly):
        designMatrix_p = designMatrixFunc2(x, y, i+1, noiseLVL)
        for j in range(int(B)):
            biasO, varianceO, MSE_train_polyO, MSE_test_polyO, randomNumber = bootStrap(designMatrix_p,z,n,scaleInp,testSize,poly,i,rn,'OLS',0, figureInp)
            bias[j] = biasO
            variance[j] = varianceO
            MSE_train_poly[j] = MSE_train_polyO
            MSE_test_poly[j] = MSE_test_polyO
            rn = randomNumber
        biasBoot[i] = np.mean(bias)
        varianceBoot[i] = np.mean(variance)
        MSE_train_boot[i] = np.mean(MSE_train_poly)
        MSE_test_boot[i] = np.mean(MSE_test_poly)

    if figureInp == True:
        MSEplot(MSE_train_boot, MSE_test_boot, poly)
        BVPlot(biasBoot, varianceBoot, MSE_test_boot, poly)


elif (str(part) == "c" or str(part) == "all"):

    # I want to check the MSE for LOOCV, 5-fold, 10-fold and 25-fold CV
    CVN = [1,5,10,25]

    # Declaring variables
    biasCV = np.zeros(len(CVN))
    varianceCV = np.zeros(len(CVN))
    MSE_train_CV = np.zeros(len(CVN))
    MSE_test_CV = np.zeros(len(CVN))
    dexxer = 0

    # Using CV to create test and train data and then calculate and plot the MSE
    for i in CVN:
        # Setting up the design matrix
        designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)
        # Performing CV (including scaling and OLS)
        bias, variance, MSE_train, MSE_test = CV(designMatrix,z,n,CVN[dexxer],scaleInp, 'OLS', 0)
        # Calculate average means for each fold
        biasCV[dexxer] = np.mean(bias)
        varianceCV[dexxer] = np.mean(variance)
        MSE_train_CV[dexxer] = np.mean(MSE_train)
        MSE_test_CV[dexxer] = np.mean(MSE_test)
        dexxer += 1


    # Now we try the Scikit-Learn version of CV
    designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)
    designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)
    rn = 0
    X_train, X_test, Y_train, Y_test, randomNumber = realSplit(designMatrix_scale, z, testSize, 0, rn, True)

    regs = LinearRegression()
    #regsCV1 = cross_val_score(regs, X_train, Y_train, scoring='neg_mean_squared_error', cv=1)
    regsCV5 = cross_val_score(regs, X_train, Y_train, scoring='neg_mean_squared_error', cv=5)
    regsCV10 = cross_val_score(regs, X_train, Y_train, scoring='neg_mean_squared_error', cv=10)
    regsCV25 = cross_val_score(regs, X_train, Y_train, scoring='neg_mean_squared_error', cv=25)

    MSE_SCI = np.array([regsCV5.mean(), regsCV10.mean(), regsCV25.mean()])

    # Plot the MSE for each fold
    if figureInp == True:
        MSEplotCV(MSE_SCI, [5,10,25], MSE_train_CV, MSE_test_CV, CVN)

elif (str(part) == "d" or str(part) == "all"):
    # Exercise 1d)
    B = input("This exercise utilizes a bootstrap resampling method that generates B sets of data of size n. How many sets should be generated? (integer)")
    bias = np.zeros(int(B))
    variance = np.zeros(int(B))
    MSE_train_poly = np.zeros(int(B))
    MSE_test_poly = np.zeros(int(B))
    biasBoot = np.zeros(poly)
    varianceBoot = np.zeros(poly)
    MSE_train_boot = np.zeros(poly)
    MSE_test_boot = np.zeros(poly)
    rn = 0
    for i in range(poly):
        designMatrix_p = designMatrixFunc2(x, y, i+1, noiseLVL)
        for j in range(int(B)):
            #print("Bootstrap iteration ", j, " out of ", int(B))
            biasO, varianceO, MSE_train_polyO, MSE_test_polyO, randomNumber = bootStrap(designMatrix_p,z,n,scaleInp,testSize,poly,i,rn,'Ridge',0)
            bias[j] = biasO
            variance[j] = varianceO
            MSE_train_poly[j] = MSE_train_polyO
            MSE_test_poly[j] = MSE_test_polyO
            rn = randomNumber
        biasBoot[i] = np.mean(bias)
        varianceBoot[i] = np.mean(variance)
        MSE_train_boot[i] = np.mean(MSE_train_poly)
        MSE_test_boot[i] = np.mean(MSE_test_poly)

    if figureInp == True:
        MSEplot(MSE_train_boot, MSE_test_boot, poly)
        BVPlot(biasBoot, varianceBoot, MSE_test_boot, poly)

elif (str(part) == "e" or str(part) == "all"):

    Y_test_pred_FINAL = np.zeros((len(z), int(len(z[0])*testSize)))
    for i in range(len(z)):
        # Setting up the polynomial design matrix with/without noise
        designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)

        # Scaling the data by subtracting mean and divide by standard deviation
        designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

        # Splitting the data into training and test sets
        rn = 0
        X_train, X_test, Y_train, Y_test, randomNumber = realSplit(designMatrix_scale, z[i,:], testSize, 0, rn, True)

        # Performing ordinary least squares (OLS)
        Y_train_pred, Y_test_pred, betas = StdPolyOLS(X_train, X_test, Y_train, Y_test)

        Y_test_pred_FINAL[i,:] = Y_test_pred

    xl = np.arange(0, 1, 1 / len(Y_test_pred_FINAL))
    yl = np.arange(0, 1, 1 / len(Y_test_pred_FINAL[0]))
    xxll, yyll = np.meshgrid(yl, xl)
    print(xxll.shape)
    print(yyll.shape)
    print(Y_test_pred_FINAL.shape)
    FrankPlot(xxll,yyll,Y_test_pred_FINAL, plot=figureInp)

elif (str(part) == "test" or str(part) == "all"):

    # Setting up the polynomial design matrix with/without noise
    designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)

    # Scaling the data by subtracting mean and divide by standard deviation
    designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

    # Splitting the data into training and test sets
    rn = 0
    X_train, X_test, Y_train, Y_test, randomNumber = realSplit2(designMatrix_scale, zz, testSize, 0, rn, True)

    # Performing ordinary least squares (OLS)
    Y_train_pred, Y_test_pred, betas = StdPolyOLS(X_train, X_test, Y_train, Y_test)

    ulti = designMatrix_scale @ betas
    FrankPlot(xx, yy, ulti, plot=figureInp)

