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
    figurePrint = input("Plot figures? (yes/no)")
    if figurePrint == "yes":
        print("Figures will be plotted")
        figureInp = True
    else:
        print("Figures will NOT be plotted")
        figureInp = False
    part = input("Which sub-exericse will you run(a,b,c,d,e,f/all)?")
    return observations, degree, scaleInp, figureInp, part


def R2(y_data, y_model):
    return 1 - (np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))

def MSE(y_data, y_pred):
    n = np.size(y_data)
    return (np.sum((y_data-y_pred)**2))/n

def FrankePlot(x, y, plot):
    # Terms of the Franke function
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    z = term1 + term2 + term3 + term4

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

    return z

def Scale(X_train,X_test, scalee):
    if scalee == True:
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_test_scale = scaler.transform(X_test)
        print("Data is scaled.")
    else:
        X_train_scale = X_train
        X_test_scale = X_test
    return X_train_scale,X_test_scale

def StdPolyOLS(X_train, X_test, Y_train, Y_test):

    # Ordinary least squares using matrix inversion
    betas = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train

    # Predict the response
    Y_train_pred = X_train @ betas
    Y_test_pred = X_test @ betas

    return Y_train_pred, Y_test_pred, betas

def designMatrixFunc(x, y, poly):
    if poly >= 2:
        p = np.arange(1,poly+1)
        preds = np.c_[x.ravel(), y.ravel()]
        for p in p:
            designMatrix = PolynomialFeatures(p).fit_transform(preds)
            params = PolynomialFeatures(p).get_params(deep=True)
    else:
        p = poly
        preds = np.c_[x.ravel(), y.ravel()]
        designMatrix = PolynomialFeatures(p).fit_transform(preds)
        params = PolynomialFeatures(p).get_params(deep=True)
    return designMatrix, params

def sciKitSplit(designMatrix,z,testsize):
    X_train, X_test, Y_train, Y_test = train_test_split(designMatrix, z, test_size=testsize)
    return X_train, X_test, Y_train, Y_test

def betaConfidenceInterval(true, n, degreeFreedom, X, betas, plot):

    sigma2 = (1 / (n - (degreeFreedom - 1))) * (sum((true - np.mean(true)) ** 2))
    # Covariance matrix
    varBeta = np.linalg.pinv((X.T @ X)) * sigma2

    # std = sqrt(var) which is the diagonal in the cov-matrix
    betaCoeff = (np.sqrt(np.diag(varBeta)))

    # Interval betas
    beta_confInt = np.c_[betas - betaCoeff, betas + betaCoeff]

    if plot == True:
        lengthX = np.arange(0, len(X[0]), 1)

        plt.xticks(lengthX)
        plt.scatter(lengthX,betas)
        plt.errorbar(lengthX,betas, yerr=beta_confInt[:, 0],fmt='none')
        plt.suptitle(r'$\beta$ -values with corresponding confidence interval', fontsize = 25)
        plt.ylabel(r'$\beta$ - value', fontsize = 20)
        plt.xlabel(r'$\beta$ - coefficients', fontsize = 20)
        plt.grid()
        plt.show()
    return beta_confInt

def CV(XX,z,n,k):
    fold = n-(n-k)
    X = np.array(XX)
    X_rep = X
    z_rep = z
    while(len(X_rep)>0):
        randRow = np.random.randint(n, size=fold)
        print(X_rep.shape)
        Xtest = X_rep[randRow,:]
        Ytest = z_rep[randRow]

        X_rep = np.delete(X_rep,int(randRow),0)
        z_rep = np.delete(z_rep, int(randRow))
        if (randRow == 0):
            Xtrain = X[(int(randRow)+1):,:]
            Ytrain = z[(int(randRow)+1):]
        elif (randRow == n):
            Xtrain = X[:int(randRow),:]
            Ytrain = z[:int(randRow)]
        else:

            Xtrain = np.concatenate((X[:int(randRow), :], X[(int(randRow)+1):, :]))
            Ytrain = np.concatenate((z[:int(randRow)], z[(int(randRow) + 1):]))

        n = n-1

    return Xtrain, Xtest, Ytrain, Ytest