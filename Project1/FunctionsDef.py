import numpy as np, scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
import scipy.stats
from numpy import array
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.patches as mpatches

def inputsss():
    observations = input("Enter number of observations n (integer): ")
    print("n = ",observations)
    degree = input("Enter the polynomial degree p (integer): ")
    print("p = ", degree)
    scalePrint = input("Scale the data? (yes/no)")
    if scalePrint == "yes":
        print("Data will be scaled")
        scaleInp = True
    else:
        print("Data will NOT be scaled")
        scaleInp = False
    noiseAR = input("Add random noise? (yes/no)")
    if noiseAR == "yes":
        print("Noise will be added")
        noiseLVL = input("With what multiplicative effect? (float)")
        noiseInp = True
    else:
        print("Noise will NOT be added")
        noiseInp = False
        noiseLVL = 0
    figurePrint = input("Plot figures? (yes/no)")
    if figurePrint == "yes":
        print("Figures will be plotted")
        figureInp = True
    else:
        print("Figures will NOT be plotted")
        figureInp = False
    part = input("Which sub-exericse will you run(a,b,c,d,e,f/all)?")
    testlevel = input("What is the test size? (between 0 and 1, but typically around 0.25)")
    return observations, degree, scaleInp, figureInp, part, noiseInp, noiseLVL,testlevel


def R2(f, y):
    return 1 - (np.sum((y - f) ** 2) / np.sum((f - np.mean(f)) ** 2))

def MSE(y_data, y_pred):
    n = np.size(y_pred)
    return (np.sum((y_data-y_pred)**2))/n

def calc_Variance(Y_test_pred,z_pred,n):
    var = (np.sum((Y_test_pred - np.mean(z_pred))**2))/np.size(Y_test_pred)
    return var

def calc_bias(z,z_pred,n):
    bias = (np.sum((z-z_pred)**2))/np.size(z)
    return bias

def FrankeFunc(x, y):
    # Terms of the Franke function
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    z = term1 + term2 + term3 + term4
    return z

def FrankPlot(x,y,z,plot):
    if plot == True:
        # Plot the Franke function
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Labels
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Color map grid
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

def FrankPlotDiff(x,y,z,z2,plot):
    if plot == True:
        # Plot the Franke function
        zz = z-z2
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Labels
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Color map grid
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

def Scale2(designMatrix, scalee):
    if scalee == True:
        designMatrix_scale = (designMatrix-np.mean(designMatrix))/ np.std(designMatrix)
    else:
        designMatrix_scale = designMatrix
    return designMatrix_scale

def StdPolyOLS(X_train, X_test, Y_train, Y_test):

    # Ordinary least squares using matrix inversion
    betas = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train

    # Predict the response
    Y_train_pred = X_train @ betas
    Y_test_pred = X_test @ betas

    #print(R2(Y_test,Y_test_pred))

    return Y_train_pred, Y_test_pred, betas

def StdPolyRidge(X_train, X_test, Y_train, Ytest, lamb):

    # Ridge using matrix inversion
    diag = np.eye(len(X_train.T @ X_train))
    betas = np.linalg.pinv((X_train.T @ X_train) + (lamb * diag)) @ X_train.T @ Y_train
    # Predict the response
    Y_train_pred = X_train @ betas
    Y_test_pred = X_test @ betas

    return Y_train_pred, Y_test_pred, betas

def designMatrixFunc2(x, y, poly, noiseLVL):

    preds = np.c_[x.ravel(), y.ravel()]
    dx = PolynomialFeatures(degree=poly, include_bias=False).fit_transform(preds)
    noise = float(noiseLVL)*np.random.randn(len(dx),len(dx[0]))
    designMatrix = dx + noise

    return designMatrix

def realSplit2(designMatrix,z,testsize,i,rn, shufflezzz):

    n = len(designMatrix)
    designMatrix2 = np.array(designMatrix)
    z2 = np.array(z)
    Xtest = np.zeros((int(testsize*n), len(designMatrix2[0])))
    Ytest = np.zeros((int(testsize*n), len(z2[0])))
    if i == 0:
        randomNumber = np.random.choice(n, size = round(testsize*n), replace = False)
    else:
        randomNumber = rn

    for i in range(len(randomNumber)):
        Xtest[i] = designMatrix2[randomNumber[i], :]
        Ytest[i] = z2[randomNumber[i],:]

    Xtrain = np.delete(designMatrix2, randomNumber, 0)
    Ytrain = np.delete(z2, randomNumber, 0)

    if i == 0:
        data = np.column_stack((Xtrain, Ytrain))
        np.random.shuffle(data)
        randomNumber2 = np.random.randint(len(data[0]),size=1)
        Ytrain = data[:,int(randomNumber2)]
        Xtrain = np.delete(data,int(randomNumber2),1)

    return Xtrain, Xtest, Ytrain, Ytest, randomNumber


def betaConfidenceInterval(true, n, degreeFreedom, X, betas, plot):

    sigma2 = np.var(true)
    # Covariance matrix
    varBeta = np.linalg.pinv((X.T @ X)) * sigma2

    # std = sqrt(var) which is the diagonal in the cov-matrix
    betaCoeff = (np.sqrt(np.diag(varBeta)))

    # Interval betas
    for i in range(len(betas[0])):
        betasR = betas[:,i]

        if plot == True:
            xxx = np.arange(0, len(betasR), 1)
            plt.xticks(xxx)
            plt.scatter(xxx,betasR)
            plt.errorbar(xxx,betasR, yerr=betaCoeff,fmt='none')
            plt.suptitle(r'$\beta$ -values with corresponding confidence interval', fontsize = 25)
            plt.ylabel(r'$\beta$ - value', fontsize = 20)
            plt.xlabel(r'$\beta$ - coefficients', fontsize = 20)
    plt.grid()
    plt.show()

    if plot == True:
        xxx = np.arange(0, len(betas[:,0]), 1)
        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.scatter(xxx, betas[:,0])
        plt.errorbar(xxx, betas[:,0], yerr=betaCoeff, fmt='none')
        plt.suptitle(r'$\beta$ -values with corresponding confidence interval', fontsize=25)
        plt.ylabel(r'$\beta$ - value', fontsize=20)
        plt.xlabel(r'$\beta$ - coefficients', fontsize=20)
        plt.grid()
        plt.show()

def bootStrap(designMatrix_poly,n,tp,lamb, figureInp, testSize,z):

    X_train_scale_poly, X_test_scale_poly, Y_train_poly, Y_test_poly, randomNumber = realSplit2(designMatrix_poly, z, testSize, 0, 0, True)

    Y_test_pred_poly = np.zeros((len(Y_test_poly),len(Y_test_poly[0])))
    Y_train_pred_poly = np.zeros((len(Y_train_poly),len(Y_train_poly[0])))
    BootstrapXTrain = np.zeros((len(X_train_scale_poly),len(X_train_scale_poly[0])))
    BootstrapXTest = np.zeros((len(X_test_scale_poly),len(X_test_scale_poly[0])))
    BootstrapYTrain = np.zeros((len(Y_train_poly),len(Y_train_poly[0])))
    BootstrapYTest = np.zeros((len(Y_test_poly),len(Y_test_poly[0])))

    for i in range(len(X_train_scale_poly)):
        BootstrapXTrain[i,:] = np.random.choice(X_train_scale_poly.ravel(), size = len(X_train_scale_poly[0]), replace = True)
        BootstrapYTrain[i,:] = np.random.choice(Y_train_poly.ravel(), size = len(Y_train_poly[0]), replace = True)
    print(len(X_train_scale_poly))
    for i in range(len(X_test_scale_poly)):
        BootstrapXTest[i, :] = np.random.choice(X_test_scale_poly.ravel(), size=len(X_test_scale_poly[0]), replace=True)
        BootstrapYTest[i,:] = np.random.choice(Y_test_poly.ravel(), size = len(Y_test_poly[0]), replace = True)

    if tp == 'OLS':
        Y_train_pred_poly, Y_test_pred_poly, betas_poly = StdPolyOLS(BootstrapXTrain, BootstrapXTest, BootstrapYTrain, BootstrapYTest)
    elif tp == 'Ridge':
        Y_train_pred_poly, Y_test_pred_poly, betas_poly = StdPolyRidge(BootstrapXTrain, BootstrapXTest, BootstrapYTrain, BootstrapYTest, lamb)
    elif tp == 'Lasso':
        print("Coming")

    bias = calc_bias(BootstrapYTest, Y_test_pred_poly, n)
    variance = calc_Variance(Y_test_pred_poly)
    MSE_train_poly = MSE(BootstrapYTrain, Y_train_pred_poly)
    MSE_test_poly = MSE(BootstrapYTest, Y_test_pred_poly)

    #x = np.arange(0, 1, 1 / n)
    #y = np.arange(0, 1, 1 / n)
    #FrankPlot(x, y, designMatrix_poly @ betas_poly, figureInp)

    return bias, variance, MSE_train_poly, MSE_test_poly, betas_poly

def CV(XX,z,n,k,scaleInp,tp,lamb,rn,p):
    # Declaring variables
    fold = int(n/k)
    z_rep = np.array(z)
    z_rep2 = np.array(z)
    bias = np.zeros(int(fold))
    variance = np.zeros(int(fold))
    MSE_train = np.zeros(int(fold))
    MSE_test = np.zeros(int(fold))
    designMatrix_rep = XX
    designMatrix_rep2 = XX
    placesss = np.zeros(k)
    betasCV = np.zeros((len(XX[0]),n,int(fold)))

    # Make sure to only shuffle the first iteration, so the test set chosen for each polynomial is the same
    for i in range(int(fold)):
        if p == 0:
            randRow = np.random.choice(len(designMatrix_rep), size=k, replace=False)
        else:
            randRow = rn

        # Choose random observations to be the test set
        Xtest = designMatrix_rep[randRow,:]
        Ytest = z_rep[randRow,:]

        # The data that is not the test set is then the training set
        for j in range(len(Xtest)):
            place = np.where((designMatrix_rep2 == Xtest[j,:]))
            placesss[j] = np.mean(place[0]).astype(int)

        Xtrain = np.delete(designMatrix_rep2, placesss.astype(int), axis=0)
        Ytrain = np.delete(z_rep2, placesss.astype(int), axis=0)

        # Deleting observations such that no observations is used as test set more than once
        z_rep2 = z
        designMatrix_rep2 = XX
        designMatrix_rep = np.delete(designMatrix_rep,randRow,axis=0)
        z_rep = np.delete(z_rep, randRow,axis=0)
        # Type of linear regression to use
        if tp == 'OLS':
            Ytrain_pred, Ytest_pred, betas = StdPolyOLS(Xtrain, Xtest, Ytrain, Ytest)
        elif tp == 'Ridge':
            Ytrain_pred, Ytest_pred, betas = StdPolyRidge(Xtrain, Xtest, Ytrain, Ytest,lamb)

        # Calculate statistics
        z_pred = XX @ betas
        bias[i] = calc_bias(Ytrain, Ytrain_pred, n)
        variance[i] = calc_Variance(Ytest_pred, z_pred, n)
        MSE_train[i] = MSE(Ytrain, Ytrain_pred)
        MSE_test[i] = MSE(Ytest, Ytest_pred)
        betasCV[:,:,i] = betas

    return bias, variance, MSE_train, MSE_test, randRow, betasCV

def MSEplot(MSE_train_poly,MSE_test_poly,poly):
    xxx = np.arange(1, poly+1, 1)
    plt.plot(xxx, MSE_train_poly, label="Train MSE", linewidth=4)
    plt.plot(xxx, MSE_test_poly, label="Test MSE", linewidth=4)
    plt.xticks(xxx)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    MSE_min_test = MSE_test_poly.argmin()
    plt.plot(xxx[MSE_min_test], MSE_test_poly[MSE_min_test], 'o', markersize=10, label="Lowest test MSE")
    plt.suptitle('Training and test MSE as a function of polynomial degree (complexity)', fontsize=25,
                 fontweight="bold")
    plt.ylabel('MSE', fontsize=20)
    plt.xlabel('Polynomial degree (complexity)', fontsize=20)
    plt.legend(loc="upper left", prop={'size': 20})
    plt.show()

def MSEplotSame(MSE_train_poly,MSE_test_poly,poly,lamb):
    xxx = np.arange(1, poly+1, 1)
    for i in range(len(lamb)):
        numsTrain = "Train MSE for lambda = " + str(lamb[i])
        numsTest = "Test MSE for lambda = " + str(lamb[i])
        plt.plot(xxx, MSE_train_poly[i,:], label=numsTrain, linewidth=4)
        plt.plot(xxx, MSE_test_poly[i,:], '--', label=numsTest, linewidth=4)
    plt.xticks(xxx)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #MSE_min_test = MSE_test_poly.argmin()
    #plt.plot(xxx[MSE_min_test], MSE_test_poly[MSE_min_test], 'o', markersize=10, label="Lowest test MSE")
    plt.suptitle('Training and test MSE as a function of model complexity and lambda', fontsize=25,
                 fontweight="bold")
    plt.ylabel('MSE', fontsize=20)
    plt.xlabel('Polynomial degree (complexity)', fontsize=20)
    plt.legend(loc="upper left", prop={'size': 15})
    plt.show()

def BVPlot(bias,variance,poly):
    xxx = np.arange(1, poly+1, 1)
    plt.plot(xxx, bias, label="Bias squared", linewidth=5)
    plt.plot(xxx, variance, label="Variance", linewidth=5)
    plt.plot(xxx, bias + variance, label="Bias + variance", linewidth=2)
    plt.xticks(xxx)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    MSE_min_test = (bias + variance).argmin()
    plt.plot(xxx[MSE_min_test], (bias + variance)[MSE_min_test], 'o', markersize=10, label="Lowest MSE")
    plt.suptitle('Bias and variance as a function of polynomial degree', fontsize=25,
                 fontweight="bold")
    plt.ylabel('Value', fontsize=20)
    plt.xlabel('Polynomial degree (complexity)', fontsize=20)
    plt.legend(loc="upper left", prop={'size': 20})
    plt.show()

def BVPlotSame(bias,variance,poly,lamb):
    xxx = np.arange(1, poly+1, 1)
    for i in range(len(lamb)):
        numsTrain = "Bias squared for lambda = " + str(lamb[i])
        numsTest = "Variance for lambda = " + str(lamb[i])
        plt.plot(xxx, bias[i,:], label=numsTrain, linewidth=4)
        plt.plot(xxx, variance[i,:], '--', label=numsTest, linewidth=4)

    #plt.plot(xxx, bias + variance, label="Bias + variance", linewidth=2)
    plt.xticks(xxx)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #MSE_min_test = (bias + variance).argmin()
    #plt.plot(xxx[MSE_min_test], (bias + variance)[MSE_min_test], 'o', markersize=10, label="Lowest MSE")
    plt.suptitle('Bias and variance as a function of polynomial degree', fontsize=25,
                 fontweight="bold")
    plt.ylabel('Value', fontsize=20)
    plt.xlabel('Polynomial degree (complexity)', fontsize=20)
    plt.legend(loc="upper left", prop={'size': 15})
    plt.show()

def BetasPlot(betas_lamb,lamb):
    xxx = np.log10(lamb)
    for i in range(len(betas_lamb)):
        for j in range(len(betas_lamb[0])):
            plt.plot(xxx, betas_lamb[i,j,:], linewidth=3)

    #plt.plot(xxx, bias + variance, label="Bias + variance", linewidth=2)
    plt.xticks(xxx)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #MSE_min_test = (bias + variance).argmin()
    #plt.plot(xxx[MSE_min_test], (bias + variance)[MSE_min_test], 'o', markersize=10, label="Lowest MSE")
    plt.suptitle('Beta values for ridge regression as function of lambda', fontsize=25, fontweight="bold")
    plt.ylabel('Beta value', fontsize=20)
    plt.xlabel('Logarithmic values of lambda', fontsize=20)
    #plt.legend(loc="upper left", prop={'size': 15})
    plt.show()

def BetasPlot2d(betas_lamb,lamb):
    xxx = np.log10(lamb)
    polyd = len(betas_lamb)
    for i in range(polyd):
        plt.plot(xxx, betas_lamb[i,:], linewidth=3)

    #plt.plot(xxx, bias + variance, label="Bias + variance", linewidth=2)
    plt.xticks(xxx)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #MSE_min_test = (bias + variance).argmin()
    #plt.plot(xxx[MSE_min_test], (bias + variance)[MSE_min_test], 'o', markersize=10, label="Lowest MSE")
    plt.suptitle('Beta values for ridge regression as function of lambda', fontsize=25, fontweight="bold")
    plt.ylabel('Beta value', fontsize=20)
    plt.xlabel('Logarithmic values of lambda', fontsize=20)
    plt.legend(loc="upper right", prop={'size': 15})
    plt.show()

def BetasPlot2d2(betas_lamb,lamb):
    xxx = np.log10(lamb)
    polyd = len(betas_lamb)
    for i in range(polyd):
        if (i<=1):
            cr = "r"
            count1 = 1
        elif (2<=i<=4):
            cr = "g"
        elif (5<=i<=8):
            cr = "b"
        elif (9<=i<=13):
            cr = "c"
        elif (14<=i<=19):
            cr = "m"
        plt.plot(xxx, betas_lamb[i,:], cr, linewidth=3)

    plt.xticks(xxx)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.suptitle('Beta values for ridge regression as function of lambda', fontsize=25, fontweight="bold")
    plt.ylabel('Beta value', fontsize=20)
    plt.xlabel('Logarithmic values of lambda', fontsize=20)
    red_patch = mpatches.Patch(color='red', label='First degree')
    green_patch = mpatches.Patch(color='green', label='Second degree')
    blue_patch = mpatches.Patch(color='blue', label='Third degree')
    cyan_patch = mpatches.Patch(color='cyan', label='Fourth degree')
    magenta_patch = mpatches.Patch(color='magenta', label='Fifth degree')

    plt.legend(handles=[red_patch, green_patch, blue_patch, cyan_patch,magenta_patch], loc="upper right", prop={'size': 15})
    plt.show()