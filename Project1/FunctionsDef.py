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

def calc_Variance(y_real, y_pred,n):
    var = np.var(y_pred)
    return var

def calc_bias(y_real, y_pred,n):
    bias = np.mean((np.mean(y_pred) - y_real)** 2) - calc_Variance(y_real,y_pred,n)
    return bias

def FrankeFunc(x, y):
    # Terms of the Franke function
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    z = term1 + term2 + term3 + term4
    return z

def FrankPlot(x,y,plot):
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

def Scale(X_train, X_test, scalee):
    if scalee == True:
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_test_scale = scaler.transform(X_test)
    else:
        X_train_scale = X_train
        X_test_scale = X_test
    return X_train_scale, X_test_scale

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

def designMatrixFunc2(x, y, poly, noiseLVL):

    preds = np.c_[x.ravel(), y.ravel()]
    designMatrix = PolynomialFeatures(degree=poly, include_bias=False).fit_transform(preds)
    data = np.column_stack((np.ones((len(designMatrix),1)), designMatrix + float(noiseLVL)*np.random.randn(len(x),len(designMatrix[0]))))

    return data

def sciKitSplit(designMatrix,z,testsize,shufflezzz):

    X_train, X_test, Y_train, Y_test = train_test_split(designMatrix, z, test_size=testsize, random_state=13, shuffle=shufflezzz)
    return X_train, X_test, Y_train, Y_test

def realSplit(designMatrix,z,testsize,i,rn, shufflezzz):

    n = len(designMatrix)

    designMatrix2 = np.array(designMatrix)
    z2 = np.array(z)
    Xtest = np.zeros((int(testsize*n), len(designMatrix2[0])))
    Ytest = np.zeros(int(testsize*n))
    if i == 0:
        randomNumber = np.random.choice(n, size = round(testsize*n), replace = False)
    else:
        randomNumber = rn

    for i in range(len(randomNumber)):
        Xtest[i] = designMatrix2[randomNumber[i], :]
        Ytest[i] = z2[randomNumber[i]]

    Xtrain = np.delete(designMatrix2, randomNumber, 0)
    Ytrain = np.delete(z2, randomNumber)

    if i == 0:
        data = np.column_stack((Xtrain, Ytrain))
        np.random.shuffle(data)
        randomNumber2 = np.random.randint(len(data[0]),size=1)
        Ytrain = data[:,int(randomNumber2)]
        Xtrain = np.delete(data,int(randomNumber2),1)
        '''''''''''
    else:
        np.random.shuffle(Xtrain)
        np.random.shuffle(Ytrain)
'''''
    return Xtrain, Xtest, Ytrain, Ytest, randomNumber




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

def bootStrap(designMatrix,z,n,scaleInp,testSize,poly,k,rn):

    Bootstrap = np.zeros((len(designMatrix),len(designMatrix[0])))
    for i in range(n):
        Bootstrap[i,:] = np.random.choice(designMatrix.ravel(), size = len(designMatrix[0]), replace = True)
    #Bootstrap = designMatrix
    designMatrix_poly = Scale2(Bootstrap, scalee=scaleInp)
    X_train_scale_poly, X_test_scale_poly, Y_train_poly, Y_test_poly, randomNumber = realSplit(designMatrix_poly, z,testSize, k, rn, True)
    rn = randomNumber
    Y_train_pred_poly, Y_test_pred_poly, betas_poly = StdPolyOLS(X_train_scale_poly, X_test_scale_poly, Y_train_poly, Y_test_poly)
    bias = calc_bias(Y_test_poly, Y_test_pred_poly, n)
    variance = calc_Variance(Y_test_poly, Y_test_pred_poly, n)
    MSE_train_poly = MSE(Y_train_poly, Y_train_pred_poly)
    MSE_test_poly = MSE(Y_test_poly, Y_test_pred_poly)

    return bias, variance, MSE_train_poly, MSE_test_poly, rn

def CV(XX,z,n,k,scaleInp):
    fold = k
    z_rep = np.array(z)
    z_rep2 = np.array(z)
    bias = np.zeros(int(len(XX)/fold))
    variance = np.zeros(int(len(XX)/fold))
    MSE_train = np.zeros(int(len(XX)/fold))
    MSE_test = np.zeros(int(len(XX)/fold))
    designMatrix = Scale2(XX, scalee=scaleInp)
    designMatrix_rep = designMatrix
    designMatrix_rep2 = designMatrix

    for i in range(int(len(designMatrix)/fold)):

        randRow = np.random.choice(len(designMatrix_rep), size=fold, replace=False)
        Xtest = designMatrix_rep[randRow,:]
        Ytest = z_rep[randRow]
        Xtrain = np.delete(designMatrix,randRow,0)
        Ytrain = np.delete(z,randRow)

        z = z_rep2
        designMatrix = designMatrix_rep2

        np.delete(designMatrix_rep,randRow,0)
        np.delete(z_rep, randRow)

        Ytrain_pred, Ytest_pred, betas = StdPolyOLS(Xtrain, Xtest, Ytrain, Ytest)
        bias[i] = calc_bias(Ytest, Ytest_pred, n)
        variance[i] = calc_Variance(Ytest, Ytest_pred, n)
        MSE_train[i] = MSE(Ytrain, Ytrain_pred)
        MSE_test[i] = MSE(Ytest, Ytest_pred)


    return bias, variance, MSE_train, MSE_test

def MSEplot(MSE_train_poly,MSE_test_poly,poly):
    xxx = np.arange(1, poly+1, 1)
    plt.plot(xxx, MSE_train_poly, label="Train MSE", linewidth=2)
    plt.plot(xxx, MSE_test_poly, label="Test MSE", linewidth=2)
    plt.xticks(xxx)
    MSE_min_test = MSE_test_poly.argmin()
    plt.plot(xxx[MSE_min_test], MSE_test_poly[MSE_min_test], 'o', markersize=10, label="Lowest test MSE")
    plt.suptitle('Training and test MSE as a function of polynomial degree (complexity)', fontsize=25,
                 fontweight="bold")
    plt.ylabel('MSE', fontsize=20)
    plt.xlabel('Polynomial degree (complexity)', fontsize=20)
    plt.legend(loc="upper left", prop={'size': 20})
    plt.show()

def MSEplotCV(regCV,CV,MSE_train_poly,MSE_test_poly,CVN):
    xxx = CVN
    plt.plot(xxx, MSE_train_poly, label="Train MSE", linewidth=2)
    plt.plot(xxx, MSE_test_poly, label="Test MSE", linewidth=2)
    plt.plot(CV, regCV, label="Test MSE (Scikit)", linewidth=2)
    plt.xticks(xxx)
    #MSE_min_test = MSE_test_poly.argmin()
    #plt.plot(xxx[MSE_min_test], MSE_test_poly[MSE_min_test], 'o', markersize=10, label="Lowest test MSE")
    plt.suptitle('Training and test MSE as a function of fold size', fontsize=25,
                 fontweight="bold")
    plt.ylabel('MSE', fontsize=20)
    plt.xlabel('Fold size', fontsize=20)
    plt.legend(loc="lower right", prop={'size': 20})
    plt.show()

def BVPlot(bias,variance,mse,poly):
    xxx = np.arange(1, poly+1, 1)
    plt.plot(xxx, bias, label="Bias", linewidth=2)
    plt.plot(xxx, variance, label="Variance", linewidth=2)
    plt.xticks(xxx)
    #bias_min = bias.argmin()
    #plt.plot(xxx[bias_min], bias[bias_min], 'o', markersize=10, label="Lowest test bias")
    plt.suptitle('Test bias and variance as a function of polynomial degree', fontsize=25,
                 fontweight="bold")
    plt.ylabel('Value', fontsize=20)
    plt.xlabel('Polynomial degree (complexity)', fontsize=20)
    plt.legend(loc="upper right", prop={'size': 20})
    plt.show()
    '''''''''
    plt.plot(xxx,bias**2 + variance, label="Bias squared + variance", linewidth=2)
    plt.plot(xxx,mse, label="MSE", linewidth=2)
    plt.plot(xxx,bias**2 + variance - mse, label="Difference", linewidth=2)
    plt.xticks(xxx)
    plt.suptitle('MSE vs. bias squared + variance as a function of polynomial degree', fontsize=25,
                 fontweight="bold")
    plt.ylabel('Value', fontsize=20)
    plt.xlabel('Polynomial degree (complexity)', fontsize=20)
    plt.legend(loc="upper right", prop={'size': 20})
    plt.show()
    '''
