import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl


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

def StdPolyOLS(poly,x,y):
    # Create the design matrix
    X = np.zeros((len(x), poly))
    for i in range(0, poly):
        X[:, i] = x ** i

    # Data split and scaling
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.75)
    standardize = StandardScaler()
    standardize.fit(X_train)
    standardize.transform(X_train)

    # Ordinary least squares using matrix inversion
    LeastSquares = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
    print("Beta coefficients for ordinary least squares: ", LeastSquares)

    # Predict the response
    Y_train_pred = X_train @ LeastSquares
    Y_test_pred = X_test @ LeastSquares

    # Calculate and print the train and test MSE
    print("Training MSE for ordinary least squares: ", MSE(Y_train, Y_train_pred), ".")
    print("Test MSE for ordinary least squares: ", MSE(Y_test, Y_test_pred), ".")
    return Y_train_pred, Y_test_pred, LeastSquares