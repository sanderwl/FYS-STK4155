import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def inputsss():
    # Input function for interactivity
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
    testlevel = input("What is the test size? (between 0 and 1, but typically around 0.25)")
    part = input("Which sub-exericse will you run(a,b,c,d,e,f/all)?")
    return 100,5,True,True,"b",True,0.001,0.25#int(observations), int(degree), bool(scaleInp), bool(figureInp), str(part), bool(noiseInp), float(noiseLVL), float(testlevel)

def inputsssA():
    # Input function for interactivity in part a)
    learn = input("Static learning rate or scheduled SGD? (constant/schedule)")
    epoch = input("Number of epochs to run? (integer, typically very large)")
    tp = input("Perform OLS or Ridge regression? (OLS/Ridge)")
    lamb = 0
    if tp == "Ridge":
        lamb = input("What is the penalty parameter? (float, typically between 0 and 1)")
    return str(learn), int(epoch), str(tp), float(lamb)

def inputsssB():
    # Input function for interactivity in part b)
    learn = input("Static learning rate or scheduled SGD? (constant/adaptive)")
    layers = input("How many layers in the network? (integer)")
    neuron = input("How many hidden neurons in the network in each layer? (integer)")
    hiddenFunc = input("What type of activation function for the hidden layers? (sigmoid, tanh, relu, leaky relu)")
    outputFunc = input("What type of activation function for the output layer? (sigmoid, tanh, relu, softmax, identity)")
    tp = input("Perform OLS or Ridge regression? (OLS/Ridge)")
    if tp == 'OLS':
        alpha = 0
    elif tp == 'Ridge':
        alpha = input("What is the penalty term in the Ridge regression? (float)")

    return 10,10,"sigmoid","constant", "identity", 0.001, "Ridge"#int(layers), int(neuron), str(hiddenFunc), str(learn), str(outputFunc), float(alpha), str(tp)

def R2(y_pred, y_real):
    # Calculate r squared
    r2 = 1 - np.sum(np.square(np.subtract(y_real, y_pred))) / np.sum(np.square(np.subtract(y_real, np.mean(y_real))))
    return r2

def MSE(y_pred, y_data):
    # Calculate MSE
    mse = np.sum(np.square(np.subtract(y_data, y_pred)))/np.size(y_pred)
    return mse

def calc_Variance(Y_test_pred,z_pred,n):
    # Calculate variance
    var = (np.sum((Y_test_pred - np.mean(z_pred))**2))/np.size(Y_test_pred)
    return var

def calc_bias(z,z_pred,n):
    # Calculate bias
    bias = (np.sum((z-z_pred)**2))/np.size(z)
    return bias

def FrankeFunc(x, y):
    # Declare terms of the Franke function
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    z = term1 + term2 + term3 + term4
    return z

def FrankeFuncNN(x, y):
    # Declare terms of the Franke function
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    z = term1 + term2 + term3 + term4
    z = z.ravel()[:, np.newaxis]
    return z

def standardize(z, scalee):
    # Standardize input by having mean equal zero and standard deviation equal one
    if scalee == True:
        z_scale = (z-np.mean(z))/ np.std(z)
    else:
        z_scale = z
    return z_scale

def normalize(z, scalee):
    # Normalize data by dividing every element by the inputs maximum value
    if scalee == True:
        z_norm = z/np.max(z)
    else:
        z_norm = z
    return z_norm

def addNoise(x, noiseLevel):
    # Add noise to whatever input
    noise = noiseLevel * np.random.randn(len(x), len(x[0]))
    output = x + noise
    return output

def createDesignmatrix(x, y, poly):
    # Creating design matrix of p polynomial degrees
    preds = np.c_[x.ravel(), y.ravel()]
    dx = PolynomialFeatures(degree=poly, include_bias=False).fit_transform(preds)
    designMatrix = np.c_[np.ones((len(dx), 1)), dx]
    return designMatrix

def createDesignmatrixNN(x, y, poly):
    # Creating design matrix of p polynomial degrees
    x1, x2 = np.meshgrid(x, x)
    p = int((poly + 1) * (poly + 2) / 2)
    X = np.ones((len(x), p))
    X = np.c_[x1.ravel()[:, np.newaxis], x2.ravel()[:, np.newaxis]]
    return X

def dataSplit(designMatrix, z, testsize, i, rn):

    n = len(designMatrix)
    designMatrix2 = np.array(designMatrix)
    z2 = np.array(z)
    Xtest = np.zeros((int(testsize*n), len(designMatrix2[0])))
    Ytest = np.zeros((int(testsize*n), len(z2[0])))
    # Random observations are found
    if i == 0:
        randomNumber = np.random.choice(n, size = round(testsize*n), replace = False)
    else:
        randomNumber = rn

    # Declare test set consisting of random number
    for i in range(len(randomNumber)):
        Xtest[i] = designMatrix2[randomNumber[i], :]
        Ytest[i] = z2[randomNumber[i],:]

    # Declare training set which is the original matrix excluding the test observations
    Xtrain = np.delete(designMatrix2, randomNumber, 0)
    Ytrain = np.delete(z2, randomNumber, 0)

    # Shuffle the data for fair split
    if i == 0:
        data = np.column_stack((Xtrain, Ytrain))
        np.random.shuffle(data)
        randomNumber2 = np.random.randint(len(data[0]),size=1)
        Ytrain = data[:,int(randomNumber2)]
        Xtrain = np.delete(data,int(randomNumber2),1)

    return Xtrain, Xtest, Ytrain, Ytest, randomNumber

def testParams(n, testSize):
    # Declare the terms when we plot the test set predictions
    x = np.arange(0, 1, 1 / (n*testSize))
    y = np.arange(0, 1, 1 / n)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def gradient(x, pred, real):
    # Gradient of OLS loss function for multi-variable input
    grad = 2 * x.T @ (pred - real)
    return grad

def gradientRidge(x, pred, real, lamb, betas):
    # Gradient of Ridge loss function for multi-variable input
    grad = 2 * x.T @ (pred - real) + 2 * lamb * betas
    return grad

def schedule(f):
    # Time based scheduling
    f0 = 5
    f1 = 50
    learningRate_new = f0 / (f1 + f)
    return learningRate_new

def SGD(epochN, designMatrix, z, tp, lamb, learningRate, learn, batch, X_test, Y_test):
    n = len(designMatrix)
    betas = np.random.randn(len(designMatrix[0]), len(z[0]))
    mse = np.zeros(epochN)
    r2 = np.zeros(epochN)
    for i in range(epochN):
        for j in range(batch):
            # One random sample per variable
            random_index = np.random.randint(n)
            Xrand = designMatrix[random_index:random_index + 1]
            Yrand = z[random_index:random_index + 1]
            # Calculate gradient of loos function
            if tp == "OLS":
                grad = gradient(Xrand, (Xrand @ betas), Yrand)
            elif tp == "Ridge":
                grad = gradientRidge(Xrand, (Xrand @ betas), Yrand, lamb, betas)
            # Set the step size based on schedule
            if learn == "constant":
                eta = learningRate
            elif learn == "schedule":
                eta = schedule(i * batch + j)
            # Step closer to minimum of the loss function
            betas = betas - eta * grad
        # Calculate MSE for each epoch iteration
        mse[i] = MSE(Y_test, X_test @ betas)
        r2[i] = R2(X_test @ betas, Y_test)
    return mse, betas, r2

