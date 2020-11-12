import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import gzip
from NeuralNetworkReg import NeuralNetwork
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPRegressor, MLPClassifier

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
    return 100,5,True,True,"d",True,0.001,0.25#int(observations), int(degree), bool(scaleInp), bool(figureInp), str(part), bool(noiseInp), float(noiseLVL), float(testlevel)

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

    return int(layers), int(neuron), str(hiddenFunc), str(learn), str(outputFunc), float(alpha), str(tp)

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

def getNumbers(nums, nums2):

    with gzip.open('C:/Users/Sander/Documents/GitHub/FYS-STK4155/Project2/Project2/Report/data/train-images-idx3-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8) \
            .reshape((image_count, row_count, column_count))
        images = images[0:nums,:,:]

    with gzip.open('C:/Users/Sander/Documents/GitHub/FYS-STK4155/Project2/Project2/Report/data/train-labels-idx1-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        labels = labels[0:nums]

    with gzip.open('C:/Users/Sander/Documents/GitHub/FYS-STK4155/Project2/Project2/Report/data/t10k-images-idx3-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_test = f.read()
        images_test = np.frombuffer(image_test, dtype=np.uint8) \
            .reshape((image_count, row_count, column_count))
        images_test = images_test[0:nums2,:,:]

    with gzip.open('C:/Users/Sander/Documents/GitHub/FYS-STK4155/Project2/Project2/Report/data/t10k-labels-idx1-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_test= f.read()
        labels_test = np.frombuffer(label_test, dtype=np.uint8)
        labels_test = labels_test[0:nums2]

    return images, labels, images_test, labels_test

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

def gradientLog(x, pred, real):
    # Gradient of OLS loss function for multi-variable input
    exp_term = np.exp(pred)
    softmax = exp_term / exp_term.sum(axis=1, keepdims=True)
    grad = 2 * x.T @ (softmax - real)
    return grad

def gradientRidgeLog(x, pred, real, lamb, betas):
    # Gradient of Ridge loss function for multi-variable input
    exp_term = np.exp(pred)
    softmax = exp_term / exp_term.sum(axis=1, keepdims=True)
    grad = 2 * x.T @ (softmax - real) + 2 * lamb * betas
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

def SGDLog(epochN, designMatrix, z, tp, lamb, learningRate, learn, batch, X_test, Y_test):
    n = len(designMatrix)
    betas = np.random.randn(len(designMatrix[0]), len(z[0]))
    acc = np.zeros(epochN)
    for i in range(epochN):
        for j in range(batch):
            # One random sample per variable
            random_index = np.random.randint(n)
            Xrand = designMatrix[random_index:random_index + 1]
            Yrand = z[random_index:random_index + 1]
            # Calculate gradient of loos function
            if tp == "OLS":
                grad = gradientLog(Xrand, (Xrand @ betas), Yrand)
            elif tp == "Ridge":
                grad = gradientRidgeLog(Xrand, (Xrand @ betas), Yrand, lamb, betas)
            # Set the step size based on schedule
            if learn == "constant":
                eta = learningRate
            elif learn == "schedule":
                eta = schedule(i * batch + j)
            # Step closer to minimum of the loss function
            betas = betas - eta * grad
        # Calculate accuracy for each epoch iteration
        acc[i] = accuracy_score(Y_test, X_test @ betas)
    return acc, betas

def findLearnRate(learningRates, alphas, hiddenFunc, hiddenFunc2, inputs, Y_train_1hot, inputs2, Y_test, layers, epochN, batch):

    AccySig = np.zeros(len(learningRates))
    AccySciSig = np.zeros(len(learningRates))
    AccyRelu = np.zeros(len(learningRates))
    AccySciRelu = np.zeros(len(learningRates))

    for k in range(len(hiddenFunc)):
        for i in range(len(learningRates)):
            print(i)
            model = NeuralNetwork(
                X=inputs,
                y=Y_train_1hot,
                hiddenNeurons=layers,
                learningRate=learningRates[i],
                batch_size=batch,
                learningType='constant',
                epochsN=epochN,
                activationType=hiddenFunc[k],
                outputFunctionType='softmax',
                cost_func_str='ce',
                alpha=0)
            model.train()
            Y_test_pred = model.predict(inputs2)
            Y_pred = model.predict_class(inputs2)

            if hiddenFunc2[k] == 'logistic':
                AccySig[i] = accuracy_score(Y_test, Y_pred)
            elif hiddenFunc2[k] == 'relu':
                AccyRelu[i] = accuracy_score(Y_test, Y_pred)

            # Scikit implementation
            dnnSci = MLPClassifier(hidden_layer_sizes=layers, activation=hiddenFunc2[k], alpha=0,
                                   learning_rate_init=learningRates[i], max_iter=epochN)
            dnnSci.fit(inputs, Y_train_1hot)
            Y_pred1 = dnnSci.predict_proba(inputs2)
            Y_pred2 = np.argmax(Y_pred1, axis=1)
            if hiddenFunc2[k] == 'logistic':
                AccySciSig[i] = accuracy_score(Y_test, Y_pred2)
            elif hiddenFunc2[k] == 'relu':
                AccySciRelu[i] = accuracy_score(Y_test, Y_pred2)

    lrateOptSig = learningRates[np.argmax(AccySig)]
    print(np.argmax(AccySig))
    print(AccySig)
    lrateOptRelu = learningRates[np.argmax(AccyRelu)]
    print(np.argmax(AccyRelu))
    print(AccyRelu)
    optRates = [lrateOptSig, lrateOptRelu]

    return optRates, AccySig, AccyRelu, AccySciSig, AccySciRelu

def findNeuronLayers(layersFind, neuronFind, hiddenFunc, hiddenFunc2, inputs, Y_train_1hot, inputs2, Y_test, layers, optRates, epochN, batch, alpha):
    AccFindSig = np.zeros((len(layersFind), len(neuronFind)))
    AccFindRelu = np.zeros((len(layersFind), len(neuronFind)))
    AccFindSciSig = np.zeros((len(layersFind), len(neuronFind)))
    AccFindSciRelu = np.zeros((len(layersFind), len(neuronFind)))

    for k in range(len(hiddenFunc)):
        for i in range(len(layersFind)):
            for j in range(len(neuronFind)):
                print("[Number of layers: ", i+1, ", number of neurons: ", j+1, "]")
                # My neural network
                model = NeuralNetwork(
                    X=inputs,
                    y=Y_train_1hot,
                    hiddenNeurons=layers,
                    learningRate=optRates[k],
                    batch_size=batch,
                    learningType='constant',
                    epochsN=epochN,
                    activationType=hiddenFunc[k],
                    outputFunctionType='softmax',
                    alpha=alpha[k])
                model.train()
                Y_test_pred_x = model.predict(inputs2)
                Y_pred_x = model.predict_class(inputs2)

                if hiddenFunc2[k] == 'logistic':
                    AccFindSig[i,j] = accuracy_score(Y_test, Y_pred_x)
                elif hiddenFunc2[k] == 'relu':
                    AccFindRelu[i,j] = accuracy_score(Y_test, Y_pred_x)

                # Scikit implementation
                dnnSci = MLPClassifier(hidden_layer_sizes=[i+1, j+1], activation=hiddenFunc2[k], alpha=alpha[k],
                                       learning_rate_init=optRates[k], max_iter=epochN)
                dnnSci.fit(inputs, Y_train_1hot)
                Y_pred1_x = dnnSci.predict_proba(inputs2)
                Y_pred2_x = np.argmax(Y_pred1_x, axis=1)
                if hiddenFunc2[k] == 'logistic':
                    AccFindSciSig[i, j] = accuracy_score(Y_test, Y_pred2_x)
                elif hiddenFunc2[k] == 'relu':
                    AccFindSciRelu[i, j] = accuracy_score(Y_test, Y_pred2_x)

    MaxAccSig = np.unravel_index(AccFindSig.argmax(), AccFindSig.shape)
    MaxAccRelu = np.unravel_index(AccFindRelu.argmax(), AccFindRelu.shape)
    optLayerSig = layersFind[MaxAccSig[0]]
    optNeuronSig = neuronFind[MaxAccSig[1]]
    optLayerRelu = layersFind[MaxAccRelu[0]]
    optNeuronRelu = neuronFind[MaxAccRelu[1]]
    optTotalSig = np.repeat(optNeuronSig, optLayerSig)
    optTotalRelu = np.repeat(optNeuronRelu, optLayerRelu)
    optTotal = [optTotalSig, optTotalRelu]
    return optTotal, AccFindSig, AccFindRelu, AccFindSciSig, AccFindSciRelu

def bestModel(optTotal, hiddenFunc, hiddenFunc2, inputs, Y_train_1hot, inputs2, Y_test, layers, optRates, epochN, batch, alpha):
    for k in range(len(hiddenFunc)):
        EpicModel = NeuralNetwork(
            X=inputs,
            y=Y_train_1hot,
            hiddenNeurons= list(optTotal[k]),
            learningRate=optRates[k],
            batch_size=batch,
            learningType='constant',
            epochsN=epochN,
            activationType=hiddenFunc[k],
            outputFunctionType='softmax',
            cost_func_str='ce',
            alpha=alpha[k])
        EpicModel.train()
        Y_test_pred_epic = EpicModel.predict(inputs2)
        Y_pred_epic = EpicModel.predict_class(inputs2)

        if hiddenFunc2[k] == 'logistic':
            AccSig_epic = accuracy_score(Y_test, Y_pred_epic)
        elif hiddenFunc2[k] == 'relu':
            AccRelu_epic = accuracy_score(Y_test, Y_pred_epic)

        # Scikit implementation
        dnnSci_epic = MLPClassifier(hidden_layer_sizes=optTotal[k], activation=hiddenFunc2[k], alpha=alpha[k],
                               learning_rate_init=optRates[k], max_iter=epochN)
        dnnSci_epic.fit(inputs, Y_train_1hot)
        Y_pred1_epic = dnnSci_epic.predict_proba(inputs2)
        Y_pred2_epic = np.argmax(Y_pred1_epic, axis=1)
        if hiddenFunc2[k] == 'logistic':
            AccSciSig_epic = accuracy_score(Y_test, Y_pred2_epic)
        elif hiddenFunc2[k] == 'relu':
            AccSciRelu_epic = accuracy_score(Y_test, Y_pred2_epic)

    return AccSig_epic, AccRelu_epic, AccSciSig_epic, AccSciRelu_epic

def learnAlpha(learningRates, alphas, hiddenFunc, hiddenFunc2, inputs, Y_train_1hot, inputs2, Y_test, layers, epochN, batch):
    AccSig_Ridge = np.zeros((len(learningRates), len(alphas)))
    AccRelu_Ridge = np.zeros((len(learningRates), len(alphas)))
    AccSciSig_Ridge = np.zeros((len(learningRates), len(alphas)))
    AccSciRelu_Ridge = np.zeros((len(learningRates), len(alphas)))
    for k in range(len(hiddenFunc)):
        for i in range(len(learningRates)):
            print(i)
            for j in range(len(alphas)):
                # My implementation
                model = NeuralNetwork(
                    X=inputs,
                    y=Y_train_1hot,
                    hiddenNeurons=layers,
                    learningRate=learningRates[i],
                    batch_size=batch,
                    learningType='constant',
                    epochsN=epochN,
                    activationType=hiddenFunc[k],
                    outputFunctionType='softmax',
                    alpha=alphas[j])
                model.train()
                Y_test_Ridge = model.predict(inputs2)
                Y_pred_Ridge = model.predict_class(inputs2)

                if hiddenFunc2[k] == 'logistic':
                    AccSig_Ridge[i, j] = accuracy_score(Y_test, Y_pred_Ridge)
                elif hiddenFunc2[k] == 'relu':
                    AccRelu_Ridge[i, j] = accuracy_score(Y_test, Y_pred_Ridge)

                # Scikit implementation
                dnnSciRidge = MLPClassifier(hidden_layer_sizes=layers, activation=hiddenFunc2[k], alpha=alphas[j],
                                            learning_rate_init=learningRates[i], max_iter=epochN)
                dnnSciRidge.fit(inputs, Y_train_1hot)

                Y_pred_Sci_Ridge = dnnSciRidge.predict_proba(inputs2)
                Y_pred_Sci_Ridge2 = np.argmax(Y_pred_Sci_Ridge, axis=1)

                if hiddenFunc2[k] == 'logistic':
                    AccSciSig_Ridge[i, j] = accuracy_score(Y_test, Y_pred_Sci_Ridge2)
                elif hiddenFunc2[k] == 'relu':
                    AccSciRelu_Ridge[i, j] = accuracy_score(Y_test, Y_pred_Sci_Ridge2)

    MaxAccSig = np.unravel_index(AccSig_Ridge.argmax(), AccSig_Ridge.shape)
    MaxAccRelu = np.unravel_index(AccRelu_Ridge.argmax(), AccRelu_Ridge.shape)
    optLrateSig = learningRates[MaxAccSig[1]]
    optLrateRelu = learningRates[MaxAccRelu[1]]
    optAlphaSig = alphas[MaxAccSig[0]]
    optAlphaRelu = alphas[MaxAccRelu[0]]
    optLrate = [optLrateSig, optLrateRelu]
    optALpha = [optAlphaSig, optAlphaRelu]
    return optLrate, optALpha, AccSig_Ridge, AccRelu_Ridge, AccSciSig_Ridge, AccSciRelu_Ridge