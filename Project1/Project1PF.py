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
from FunctionsDef import inputsss, FrankeFunc, MSE, R2,StdPolyOLS, designMatrixFunc2,\
    betaConfidenceInterval, MSEplot, calc_Variance, calc_bias, BVPlot, Scale2,FrankPlot,\
    bootStrap, CV, realSplit2, FrankPlotDiff, StdPolyRidge, MSEplotSame, BVPlotSame, BetasPlot, \
    BetasPlot2d, BetasPlot2d2, terrainLoad, terrainPlot, designMatrixFunc3
import scipy.stats
from mlxtend.evaluate import bias_variance_decomp
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from mlxtend.evaluate import bias_variance_decomp
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

# Input function
observations, degree, scaleInp, figureInp, part, noiseInp, noiseLVL,testsizesss = inputsss()

# Test size throughout script
testSize = float(testsizesss)

# Create data
n = int(observations)
poly = int(degree)
x = np.arange(0, 1, 1 / n)
y = np.arange(0, 1, 1 / n)
xx, yy = np.meshgrid(x, y)

xt = np.arange(0, 1, 1 / (n*testSize))
xxtt, yytt = np.meshgrid(y,xt)

# Setting up the Franke function with/without noise
z = FrankeFunc(xx, yy) #+ float(noiseLVL) * np.random.randn(n)

if (str(part) == "a" or str(part) == "all"):

    # Exercise 1a)
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

    #FrankPlot(xxtt,yytt,Y_test_pred,plot=figureInp)


elif (str(part) == "b" or str(part) == "all"):
    # Exercise 1b)
    B = input("This exercise utilizes a bootstrap resampling method that generates B sets of data of size n. How many sets should be generated? (integer)")
    if (int(B) > 0):
        MSE_train = np.zeros(int(B))
        MSE_test = np.zeros(int(B))
        bias = np.zeros(int(B))
        variance = np.zeros(int(B))
        MSE_train_boot = np.zeros(poly)
        MSE_test_boot = np.zeros(poly)
        bias_boot = np.zeros(poly)
        variance_boot = np.zeros(poly)
        rn = 0

        for k in range(poly):
            print("p = ", k)
            # Setting up the polynomial design matrix with/without noise
            designMatrix = designMatrixFunc2(x, y, k+1, noiseLVL)

            # Scaling the data by subtracting mean and divide by standard deviation
            designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

            designMatrixBoot = np.zeros((len(designMatrix), len(designMatrix[0])))
            estimated = np.zeros((n, n, int(B)))

            for i in range(int(B)):
                for j in range(len(designMatrix_scale[0])):
                    designMatrixBoot[:, j] = np.random.choice(designMatrix_scale[:,j], size=len(designMatrix), replace=True)

                # Splitting the data into training and test sets
                X_train, X_test, Y_train, Y_test, randomNumber = realSplit2(designMatrixBoot, z, testSize, 0, rn, True)
                rn = randomNumber
                # Performing ordinary least squares (OLS)
                Y_train_pred, Y_test_pred, betas = StdPolyOLS(X_train, X_test, Y_train, Y_test)

                z_pred = designMatrix_scale @ betas

                estimated[:, :, i] = designMatrixBoot @ betas
                MSE_train[i] = MSE(Y_train, Y_train_pred)
                MSE_test[i] = MSE(Y_test, Y_test_pred)
                bias[i] = calc_bias(Y_train, Y_train_pred,n)
                variance[i] = calc_Variance(Y_test_pred,z_pred,n)

            MSE_train_boot[k] = np.mean(MSE_train)
            MSE_test_boot[k] = np.mean(MSE_test)
            bias_boot[k] = np.mean(bias)
            variance_boot[k] = np.mean(variance)

            # Plotting the estimated Franke function
            #estimatedBoot = np.mean(estimated, 2)
            #FrankPlot(xx, yy, estimatedBoot, plot=figureInp)

        if figureInp == True:
            MSEplot(MSE_train_boot,MSE_test_boot,poly)
            BVPlot(bias_boot,variance_boot,poly)
    else:
        print("B must be larger than 0!")

elif (str(part) == "c" or str(part) == "all"):

    # I want to check the MSE for LOOCV, 5-fold, 10-fold and 25-fold CV
    CVN = int(input("Here we aim to use the cross-validation resampling method. What is the fold size? (integer between 1 and n/2)"))

    # Declaring variables
    biasCV = np.zeros(poly)
    varianceCV = np.zeros(poly)
    MSE_train_CV = np.zeros(poly)
    MSE_test_CV = np.zeros(poly)
    rn = 0

    for i in range(poly):
        # Setting up the design matrix
        designMatrix = designMatrixFunc2(x, y, i + 1, noiseLVL)

        # Scaling the data by subtracting mean and divide by standard deviation
        designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

        # Performing CV with OLS
        bias, variance, MSE_train, MSE_test, randRow = CV(designMatrix_scale, z, n, CVN, scaleInp, 'OLS', 0, rn, i)
        rn = randRow
        # Calculate average means for each fold
        biasCV[i] = np.mean(bias)
        varianceCV[i] = np.mean(variance)
        MSE_train_CV[i] = np.mean(MSE_train)
        MSE_test_CV[i] = np.mean(MSE_test)

    # Plot the MSE for each fold
    if figureInp == True:
        MSEplot(MSE_train_CV, MSE_test_CV, poly)
        BVPlot(biasCV, varianceCV, poly)


elif (str(part) == "d" or str(part) == "all"):
    bc = input("Bootstrap or CV? (b/cv)")
    ye = input("Investigate as function of complexity or as function of lambda? (c or l)")
    # Exercise 1d)
    if bc == "b":
        B = input("This exercise utilizes a bootstrap resampling method that generates B sets of data of size n. How many sets should be generated? (integer)")
        if ye == "c":
            lamb = [0.001, 0.01, 0.1, 1, 10]
            MSE_train = np.zeros(int(B))
            MSE_test = np.zeros(int(B))
            bias = np.zeros(int(B))
            variance = np.zeros(int(B))
            MSE_train_boot = np.zeros(poly)
            MSE_test_boot = np.zeros(poly)
            MSE_train_lamb = np.zeros((len(lamb),poly))
            MSE_test_lamb = np.zeros((len(lamb),poly))
            bias_boot = np.zeros(poly)
            variance_boot = np.zeros(poly)
            bias_lamb = np.zeros((len(lamb),poly))
            variance_lamb = np.zeros((len(lamb),poly))

            rn = 0
            dexxer = 0

            for f in lamb:
                print(dexxer)
                for k in range(poly):
                    # Setting up the polynomial design matrix with/without noise
                    designMatrix = designMatrixFunc2(x, y, k + 1, noiseLVL)

                    # Scaling the data by subtracting mean and divide by standard deviation
                    designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                    designMatrixBoot = np.zeros((len(designMatrix), len(designMatrix[0])))
                    estimated = np.zeros((n, n, int(B)))

                    for i in range(int(B)):
                        for j in range(len(designMatrix_scale[0])):
                            designMatrixBoot[:, j] = np.random.choice(designMatrix_scale[:, j], size=len(designMatrix), replace=True)


                        # Splitting the data into training and test sets
                        X_train, X_test, Y_train, Y_test, randomNumber = realSplit2(designMatrixBoot, z, testSize, 0, rn, True)
                        rn = randomNumber
                        # Performing ordinary least squares (OLS)
                        Y_train_pred, Y_test_pred, betas = StdPolyRidge(X_train, X_test, Y_train, Y_test,f)

                        z_pred = designMatrix_scale @ betas
                        estimated[:, :, i] = designMatrixBoot @ betas
                        MSE_train[i] = MSE(Y_train, Y_train_pred)
                        MSE_test[i] = MSE(Y_test, Y_test_pred)
                        bias[i] = calc_bias(Y_train, Y_train_pred, n)
                        variance[i] = calc_Variance(Y_test_pred, z_pred, n)

                    MSE_train_boot[k] = np.mean(MSE_train)
                    MSE_test_boot[k] = np.mean(MSE_test)
                    bias_boot[k] = np.mean(bias)
                    variance_boot[k] = np.mean(variance)

                MSE_train_lamb[dexxer,:] = MSE_train_boot
                MSE_test_lamb[dexxer,:] = MSE_test_boot
                bias_lamb[dexxer,:] = bias_boot
                variance_lamb[dexxer,:] = variance_boot

                dexxer += 1

            if figureInp == True:
                MSEplotSame(MSE_train_lamb,MSE_test_lamb,poly,lamb)
                BVPlotSame(bias_lamb,variance_lamb,poly,lamb)
        else:
            lamb = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

            rn = 0
            dexxer = 0
            designMatrixxx = designMatrixFunc2(x, y, poly, noiseLVL)
            betas_boot = np.zeros((len(designMatrixxx[0]), n, int(B)))
            betas_lamb = np.zeros((len(designMatrixxx[0]), n, len(lamb)))

            for f in lamb:
                print(dexxer)
                # Setting up the polynomial design matrix with/without noise
                designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)

                # Scaling the data by subtracting mean and divide by standard deviation
                designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                designMatrixBoot = np.zeros((len(designMatrix), len(designMatrix[0])))

                for i in range(int(B)):
                    for j in range(len(designMatrix_scale[0])):
                        designMatrixBoot[:, j] = np.random.choice(designMatrix_scale[:, j], size=len(designMatrix), replace=True)

                    # Splitting the data into training and test sets
                    X_train, X_test, Y_train, Y_test, randomNumber = realSplit2(designMatrixBoot, z, testSize, 0, rn, True)
                    rn = randomNumber
                    # Performing ordinary least squares (OLS)
                    Y_train_pred, Y_test_pred, betas = StdPolyRidge(X_train, X_test, Y_train, Y_test, f)

                    betas_boot[:, :, i] = betas

                betas_lamb[:, :, dexxer] = np.mean(betas_boot, axis=2)
                dexxer += 1
            betascool = np.zeros((len(betas_lamb[0]), len(lamb)))
            # for q in range(len(betas_lamb[0])):
            # betascool[q, :] = betas_lamb[q,q,:]
            betascool = betas_lamb[:, 5, :]

            if figureInp == True:
                BetasPlot2d(betascool, lamb)
                BetasPlot2d2(betascool, lamb)
    else:
        if ye == "c":
            # I want to check the MSE for LOOCV, 5-fold, 10-fold and 25-fold CV
            CVN = int(input("Here we aim to use the cross-validation resampling method. What is the fold size? (integer between 1 and n/2)"))

            # Declaring variables
            biasCV = np.zeros(poly)
            varianceCV = np.zeros(poly)
            MSE_train_CV = np.zeros(poly)
            MSE_test_CV = np.zeros(poly)
            lamb = [0.001, 0.01, 0.1, 1, 10]
            MSE_train_fin = np.zeros((len(lamb),poly))
            MSE_test_fin = np.zeros((len(lamb),poly))
            bias_fin = np.zeros((len(lamb),poly))
            variance_fin = np.zeros((len(lamb),poly))
            rn = 0
            dexxer = 0
            for f in lamb:
                print(dexxer)
                for i in range(poly):
                    # Setting up the design matrix
                    designMatrix = designMatrixFunc2(x, y, i + 1, noiseLVL)

                    # Scaling the data by subtracting mean and divide by standard deviation
                    designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                    # Performing CV with OLS
                    bias, variance, MSE_train, MSE_test, randRow, betas = CV(designMatrix_scale, z, n, CVN, scaleInp, 'Ridge', f, rn, i)
                    rn = randRow
                    # Calculate average means for each fold
                    biasCV[i] = np.mean(bias)
                    varianceCV[i] = np.mean(variance)
                    MSE_train_CV[i] = np.mean(MSE_train)
                    MSE_test_CV[i] = np.mean(MSE_test)

                MSE_train_fin[dexxer,:] = MSE_train_CV
                MSE_test_fin[dexxer,:] = MSE_test_CV
                bias_fin[dexxer,:] = biasCV
                variance_fin[dexxer,:] = varianceCV
                dexxer += 1

            if figureInp == True:
                MSEplotSame(MSE_train_fin,MSE_test_fin,poly,lamb)
                BVPlotSame(bias_fin,variance_fin,poly,lamb)

        if ye == "l":
            # I want to check the MSE for LOOCV, 5-fold, 10-fold and 25-fold CV
            CVN = int(input("Here we aim to use the cross-validation resampling method. What is the fold size? (integer between 1 and n/2)"))

            # Declaring variables

            lamb = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

            designMatrixxxx = designMatrixFunc2(x, y, poly, noiseLVL)
            betasCV = np.zeros((len(designMatrixxxx[0]),n,len(lamb)))
            betas_fin = np.zeros((len(designMatrixxxx[0]),n,len(lamb)))
            rn = 0
            dexxer = 0

            for f in lamb:
                print(dexxer)
                # Setting up the design matrix
                designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)

                # Scaling the data by subtracting mean and divide by standard deviation
                designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                # Performing CV with Ridge
                bias, variance, MSE_train, MSE_test, randRow, betas = CV(designMatrix_scale, z, n, CVN, scaleInp, 'Ridge', f, rn, 0)
                rn = randRow
                # Calculate mean betas
                betasCV[:,:,dexxer] = np.mean(betas, axis=2)

                dexxer += 1
            betas_fin = betasCV[:, 5, :]
            if figureInp == True:
                BetasPlot2d(betas_fin, lamb)
                BetasPlot2d2(betas_fin, lamb)


elif (str(part) == "e" or str(part) == "all"):

    bc = input("Bootstrap or CV? (b/cv)")
    ye = input("Investigate as function of complexity or as function of lambda? (c or l)")
    # Exercise 1d)
    if bc == "b":
        B = input("This exercise utilizes a bootstrap resampling method that generates B sets of data of size n. How many sets should be generated? (integer)")
        if ye == "c":
            lamb = [0.0001, 0.001, 0.01, 0.1, 0]
            MSE_train = np.zeros(int(B))
            MSE_test = np.zeros(int(B))
            bias = np.zeros(int(B))
            variance = np.zeros(int(B))
            MSE_train_boot = np.zeros(poly)
            MSE_test_boot = np.zeros(poly)
            MSE_train_lamb = np.zeros((len(lamb),poly))
            MSE_test_lamb = np.zeros((len(lamb),poly))
            bias_boot = np.zeros(poly)
            variance_boot = np.zeros(poly)
            bias_lamb = np.zeros((len(lamb),poly))
            variance_lamb = np.zeros((len(lamb),poly))

            rn = 0
            dexxer = 0

            for f in lamb:
                print(dexxer)
                for k in range(poly):
                    # Setting up the polynomial design matrix with/without noise
                    designMatrix = designMatrixFunc2(x, y, k + 1, noiseLVL)

                    # Scaling the data by subtracting mean and divide by standard deviation
                    designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                    designMatrixBoot = np.zeros((len(designMatrix), len(designMatrix[0])))
                    estimated = np.zeros((n, n, int(B)))

                    for i in range(int(B)):
                        for j in range(len(designMatrix_scale[0])):
                            designMatrixBoot[:, j] = np.random.choice(designMatrix_scale[:, j], size=len(designMatrix), replace=True)


                        # Splitting the data into training and test sets
                        X_train, X_test, Y_train, Y_test, randomNumber = realSplit2(designMatrixBoot, z, testSize, 0, rn, True)
                        rn = randomNumber
                        # Performing Lasso regression
                        reg = Lasso(alpha=f, random_state=0, fit_intercept=False)
                        reg.fit(X_train, Y_train)
                        betas = reg.coef_.T

                        Y_train_pred = reg.predict(X_train)
                        Y_test_pred = reg.predict(X_test)
                        z_pred = designMatrixBoot @ betas

                        estimated[:, :, i] = designMatrixBoot @ betas
                        MSE_train[i] = MSE(Y_train, Y_train_pred)
                        MSE_test[i] = MSE(Y_test, Y_test_pred)
                        bias[i] = calc_bias(Y_train, Y_train_pred, n)
                        variance[i] = calc_Variance(Y_test_pred, z_pred, n)

                    MSE_train_boot[k] = np.mean(MSE_train)
                    MSE_test_boot[k] = np.mean(MSE_test)
                    bias_boot[k] = np.mean(bias)
                    variance_boot[k] = np.mean(variance)

                MSE_train_lamb[dexxer,:] = MSE_train_boot
                MSE_test_lamb[dexxer,:] = MSE_test_boot
                bias_lamb[dexxer,:] = bias_boot
                variance_lamb[dexxer,:] = variance_boot

                dexxer += 1

            if figureInp == True:
                MSEplotSame(MSE_train_lamb,MSE_test_lamb,poly,lamb)
                BVPlotSame(bias_lamb,variance_lamb,poly,lamb)
        else:
            lamb = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

            rn = 0
            dexxer = 0
            designMatrixxx = designMatrixFunc2(x, y, poly, noiseLVL)
            betas_boot = np.zeros((len(designMatrixxx[0]), n, int(B)))
            betas_lamb = np.zeros((len(designMatrixxx[0]), n, len(lamb)))

            for f in lamb:
                print(dexxer)
                # Setting up the polynomial design matrix with/without noise
                designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)

                # Scaling the data by subtracting mean and divide by standard deviation
                designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                designMatrixBoot = np.zeros((len(designMatrix), len(designMatrix[0])))

                for i in range(int(B)):
                    for j in range(len(designMatrix_scale[0])):
                        designMatrixBoot[:, j] = np.random.choice(designMatrix_scale[:, j], size=len(designMatrix), replace=True)

                    # Splitting the data into training and test sets
                    X_train, X_test, Y_train, Y_test, randomNumber = realSplit2(designMatrixBoot, z, testSize, 0, rn, True)
                    rn = randomNumber
                    # Performing Lasso regression
                    reg = Lasso(alpha=f, random_state=0, fit_intercept=False)
                    reg.fit(X_train, Y_train)
                    betas = reg.coef_.T

                    Ytrain_pred = reg.predict(X_train)
                    Ytest_pred = reg.predict(X_test)

                    betas_boot[:, :, i] = betas

                betas_lamb[:, :, dexxer] = np.mean(betas_boot, axis=2)
                dexxer += 1
            betascool = np.zeros((len(betas_lamb[0]), len(lamb)))
            # for q in range(len(betas_lamb[0])):
            # betascool[q, :] = betas_lamb[q,q,:]
            betascool = betas_lamb[:, 5, :]

            if figureInp == True:
                BetasPlot2d(betascool, lamb)
                BetasPlot2d2(betascool, lamb)
    else:
        if ye == "c":
            # I want to check the MSE for LOOCV, 5-fold, 10-fold and 25-fold CV
            CVN = int(input("Here we aim to use the cross-validation resampling method. What is the fold size? (integer between 1 and n/2)"))

            # Declaring variables
            biasCV = np.zeros(poly)
            varianceCV = np.zeros(poly)
            MSE_train_CV = np.zeros(poly)
            MSE_test_CV = np.zeros(poly)
            lamb = [0.0001, 0.001, 0.01, 0.1, 0]
            MSE_train_fin = np.zeros((len(lamb),poly))
            MSE_test_fin = np.zeros((len(lamb),poly))
            bias_fin = np.zeros((len(lamb),poly))
            variance_fin = np.zeros((len(lamb),poly))
            rn = 0
            dexxer = 0
            for f in lamb:
                print(dexxer)
                for i in range(poly):
                    # Setting up the design matrix
                    designMatrix = designMatrixFunc2(x, y, i + 1, noiseLVL)

                    # Scaling the data by subtracting mean and divide by standard deviation
                    designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                    # Performing CV with OLS
                    bias, variance, MSE_train, MSE_test, randRow, betas = CV(designMatrix_scale, z, n, CVN, scaleInp, 'Lasso', f, rn, i)
                    rn = randRow
                    # Calculate average means for each fold
                    biasCV[i] = np.mean(bias)
                    varianceCV[i] = np.mean(variance)
                    MSE_train_CV[i] = np.mean(MSE_train)
                    MSE_test_CV[i] = np.mean(MSE_test)

                MSE_train_fin[dexxer,:] = MSE_train_CV
                MSE_test_fin[dexxer,:] = MSE_test_CV
                bias_fin[dexxer,:] = biasCV
                variance_fin[dexxer,:] = varianceCV
                dexxer += 1

            if figureInp == True:
                MSEplotSame(MSE_train_fin,MSE_test_fin,poly,lamb)
                BVPlotSame(bias_fin,variance_fin,poly,lamb)

        if ye == "l":
            # I want to check the MSE for LOOCV, 5-fold, 10-fold and 25-fold CV
            CVN = int(input("Here we aim to use the cross-validation resampling method. What is the fold size? (integer between 1 and n/2)"))

            # Declaring variables

            lamb = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

            designMatrixxxx = designMatrixFunc2(x, y, poly, noiseLVL)
            betasCV = np.zeros((len(designMatrixxxx[0]),n,len(lamb)))
            betas_fin = np.zeros((len(designMatrixxxx[0]),n,len(lamb)))
            rn = 0
            dexxer = 0

            for f in lamb:
                print(dexxer)
                # Setting up the design matrix
                designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)

                # Scaling the data by subtracting mean and divide by standard deviation
                designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                # Performing CV with Ridge
                bias, variance, MSE_train, MSE_test, randRow, betas = CV(designMatrix_scale, z, n, CVN, scaleInp, 'Lasso', f, rn, 0)
                rn = randRow
                # Calculate mean betas
                betasCV[:,:,dexxer] = np.mean(betas, axis=2)

                dexxer += 1
            betas_fin = betasCV[:, 5, :]
            if figureInp == True:
                BetasPlot2d(betas_fin, lamb)
                BetasPlot2d2(betas_fin, lamb)



elif (str(part) == "f" or str(part) == "g" or str(part) == "all"):

    zz = terrainLoad()
    z = zz[1800:,:]
    #terrainPlot(figureInp,z)
    lenx = len(z)

    x = np.linspace(0, 1, lenx)
    y = np.linspace(0, 1, lenx)

    skip = int(10)
    z = z[::skip,::skip]
    x = x[::skip]
    y = y[::skip]

    n = len(z)
    xx, yy = np.meshgrid(x, y)

    tp = input('What regression scheme should be employed? (OLS/Ridge/Lasso)')
    ye = input("Investigate as function of complexity or as function of lambda? (c or l)")
    # I want to check the MSE for LOOCV, 5-fold, 10-fold and 25-fold CV
    CVN = int(input("Here we aim to use the cross-validation resampling method. What is the fold size? (integer between 1 and n/2)"))

    if ye == 'c':

        # Declaring variables
        biasCV = np.zeros(poly)
        varianceCV = np.zeros(poly)
        MSE_train_CV = np.zeros(poly)
        MSE_test_CV = np.zeros(poly)
        lamb = [0.000000001,0.00000001,0.0000001, 0.000001,0.00001]
        MSE_train_fin = np.zeros((len(lamb), poly))
        MSE_test_fin = np.zeros((len(lamb), poly))
        bias_fin = np.zeros((len(lamb), poly))
        variance_fin = np.zeros((len(lamb), poly))
        val = np.zeros(len(lamb))
        idx = np.zeros(len(lamb))
        rn = 0
        dexxer = 0
        for f in lamb:
            print(dexxer)
            for i in range(poly):
                # Setting up the design matrix
                designMatrix = designMatrixFunc2(x, y, i + 1, noiseLVL)

                # Scaling the data by subtracting mean and divide by standard deviation
                designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

                # Performing CV with OLS
                bias, variance, MSE_train, MSE_test, randRow, betas = CV(designMatrix_scale, z, n, CVN, scaleInp,tp, f, rn, dexxer)
                rn = randRow
                # Calculate average means for each fold
                biasCV[i] = np.mean(bias)
                varianceCV[i] = np.mean(variance)
                MSE_train_CV[i] = np.mean(MSE_train)
                MSE_test_CV[i] = np.mean(MSE_test)

            MSE_train_fin[dexxer, :] = MSE_train_CV
            MSE_test_fin[dexxer, :] = MSE_test_CV
            bias_fin[dexxer, :] = biasCV
            variance_fin[dexxer, :] = varianceCV
            val[dexxer], idx[dexxer] = min((val, idx) for (idx, val) in enumerate(MSE_test_fin[dexxer, :]))
            dexxer += 1
        print(val)
        print(idx)

        if figureInp == True:
            MSEplotSame(MSE_train_fin, MSE_test_fin, poly, lamb)
            BVPlotSame(bias_fin, variance_fin, poly, lamb)


    elif ye == 'l':
        # Declaring variables
        #lamb = [0.0000001, 0.000001,0.00001, 0.0001, 0.001, 0.01]
        lamb = [0.000000001,0.00000001,0.0000001, 0.000001,0.00001]
        designMatrixxxx = designMatrixFunc2(x, y, poly, noiseLVL)
        betasCV = np.zeros((len(designMatrixxxx[0]), n, len(lamb)))
        betas_fin = np.zeros((len(designMatrixxxx[0]), n, len(lamb)))
        rn = 0
        dexxer = 0

        for f in lamb:
            print(dexxer)
            # Setting up the design matrix
            designMatrix = designMatrixFunc2(x, y, poly, noiseLVL)

            # Scaling the data by subtracting mean and divide by standard deviation
            designMatrix_scale = Scale2(designMatrix, scalee=scaleInp)

            # Performing CV with Ridge
            bias, variance, MSE_train, MSE_test, randRow, betas = CV(designMatrix_scale, z, n, CVN, scaleInp, tp,f, rn, dexxer)
            rn = randRow
            # Calculate mean betas
            betasCV[:, :, dexxer] = np.mean(betas, axis=2)

            dexxer += 1


        if figureInp == True:
            zz = betasCV[:,:,0]
            zz_pred = designMatrix_scale @ zz
            terrainPlot(figureInp,zz_pred)
            terrainPlot(figureInp,z)


