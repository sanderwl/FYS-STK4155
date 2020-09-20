import numpy as np, scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
import scipy.stats


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

def Scale(X_train,X_test):
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)
    print("Data is scaled.")
    return X_train_scale,X_test_scale

def StdPolyOLS(X,z,scalee):

    # Data split
    X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size=0.2)

    # Scaling the data if user wants
    if scalee == True:
        X_train_scale,X_test_scale = Scale(X_train,X_test)
    else:
        X_train_scale = X_train
        X_test_scale = X_test

    # Ordinary least squares using matrix inversion
    betas = np.linalg.pinv(X_train_scale.T @ X_train_scale) @ X_train_scale.T @ Y_train

    # Predict the response
    Y_train_pred = X_train_scale @ betas
    Y_test_pred = X_test_scale @ betas

    # Calculate and print the train and test MSE
    print("Training MSE for ordinary least squares: ", MSE(Y_train, Y_train_pred), ".")
    print("Test MSE for ordinary least squares: ", MSE(Y_test, Y_test_pred), ".")

    # Calculate and print train and test R2
    print("Training R2 for ordinary least squares: ", R2(Y_train, Y_train_pred), ".")
    print("Test R2 for ordinary least squares: ", R2(Y_test, Y_test_pred), ".")

    return Y_train_pred, Y_test_pred, betas

def designMatrix(x,y,poly):

    p = np.arange(1,poly+1,1)
    preds = np.c_[x.ravel(), y.ravel()]
    for p in p:
        designMatrix = PolynomialFeatures(p).fit_transform(preds)
        params = PolynomialFeatures(p).get_params(deep=True)

    return designMatrix, params

def betaConfidenceInterval(true, n, degreeFreedom, X, betas, plot):

    sigma2 = (1 / (n - (degreeFreedom - 1))) * (sum((true - np.mean(true)) ** 2))
    # Covariance matrix
    varBeta = np.linalg.pinv((X.T @ X)) * sigma2

    # std = sqrt(var) which is the diagonal in the cov-matrix
    betaCoeff = (np.sqrt(np.diag(varBeta)))
    # Intervall betacoefisienter:
    beta_confInt = np.c_[betas - betaCoeff, betas + betaCoeff]

    if plot == True:
        xlab = [r'$\beta_0$',r'$\beta_1$',r'$\beta_2$',r'$\beta_3$',r'$\beta_4$',r'$\beta_5$',r'$\beta_6$',r'$\beta_7$',
                r'$\beta_8$',r'$\beta_9$',r'$\beta_{10}$',r'$\beta_{11}$',r'$\beta_{12}$',r'$\beta_{13}$',r'$\beta_{14}$',
                r'$\beta_{15}$',r'$\beta_{16}$',r'$\beta_{17}$',r'$\beta_{18}$',r'$\beta_{19}$',r'$\beta_{20}$']
        fig, ax = plt.subplots()
        plt.scatter(xlab,betas)
        plt.errorbar(xlab, betas, yerr=beta_confInt[:, 0],fmt='none')
        plt.suptitle(r'$\beta$ -values with corresponding confidence interval', fontsize = 25)
        plt.ylabel(r'$\beta$ -value', fontsize = 20)
        plt.xlabel('Coefficients', fontsize = 20)
        plt.grid()
        plt.show()


    return beta_confInt