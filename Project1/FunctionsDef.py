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
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data, y_pred):
    n = np.size(y_data)
    return (np.sum((y_data-y_pred)**2))/n

def FrankePlot(x, y):
    # Terms of the Franke function
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    z = term1 + term2 + term3 + term4

    # Plot the Franke function
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Labels
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Color map grid
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    return z

def Scale(x):
    standardize = StandardScaler()
    standardize.fit(x)
    d = standardize.transform(x)
    return d

def StdPolyOLS(X,z):

    # Data split and scaling
    X_train, X_test, Y_train, Y_test = train_test_split(X, z, test_size=0.2)
    X_train_scale = Scale(X_train)
    X_test_scale = Scale(X_test)

    # Ordinary least squares using matrix inversion
    betas = np.linalg.pinv(X_train_scale.T @ X_train_scale) @ X_train_scale.T @ Y_train
    print("Beta coefficients for ordinary least squares: ", betas)

    # Predict the response
    Y_train_pred = X_train @ betas
    Y_test_pred = X_test @ betas

    # Calculate and print the train and test MSE
    print("Training MSE for ordinary least squares: ", MSE(Y_train, Y_train_pred), ".")
    print("Test MSE for ordinary least squares: ", MSE(Y_test, Y_test_pred), ".")
    return Y_train_pred, Y_test_pred, betas

def designMatrix(x,y,poly):
    p = np.arange(1,poly+1,1)
    preds = np.c_[x.ravel(), y.ravel()]
    for p in p:
        designMatrix = PolynomialFeatures(p).fit_transform(preds)

    return designMatrix

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h