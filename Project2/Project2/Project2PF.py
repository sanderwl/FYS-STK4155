import numpy as np
from FunctionsDef import inputsss, FrankeFunc, standardize, createDesignmatrix, dataSplit, MSE, R2, gradient, \
    inputsssA, testParams, normalize, SGD, addNoise, inputsssB
from PlotFunctionsDef import FrankPlot, FrankPlotDiff, MSESGD, MSEvsLRATE, R2vsLRATE, MSESGDSTANDARD, heatmap
from NeuralNetworkReg import NeuralNetwork
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Input function
observations, degree, scaleInp, figureInp, part, noiseInp, noiseLVL, testsizesss = inputsss()

# Test size throughout script
testSize = testsizesss

# Create data
n = observations
poly = degree
rn = 0
x = np.arange(0, 1, 1 / n)
y = np.arange(0, 1, 1 / n)
xx, yy = np.meshgrid(x, y)

# Setting up the Franke function with/without noise
zz = FrankeFunc(xx, yy)
zz = standardize(zz, scalee=scaleInp)
zz = normalize(zz, scalee=scaleInp)
zz = addNoise(zz, noiseLevel=noiseLVL)
z = zz

if (str(part) == "a" or str(part) == "all"):

    # Exercise 1a)

    # Input values
    learn, epochN, tp, lamb = inputsssA()

    # Setting up the polynomial design matrix with/without noise
    designMatrix = createDesignmatrix(x, y, poly)

    # Splitting the data into training and test sets
    X_train, X_test, Y_train, Y_test, randomNumber = dataSplit(designMatrix, z, testSize, 0, rn)
    rn = randomNumber

    # Stochastic gradient descent to find the regression coefficients using standard parameters
    lRate = 0.01
    batch = int(n / 10)
    mseStand, betas = SGD(epochN, X_train, Y_train, tp, lamb, lRate, learn, int(batch), X_test, Y_test)

    # Stochastic gradient descent to find the regression coefficients for different batch sizes
    batchIT = [n, n / 10, n / n]
    mseBatch = np.zeros((len(batchIT), epochN))
    for i in range(len(batchIT)):
        mseBatch[i, :], betas = SGD(epochN, X_train, Y_train, tp, lamb, lRate, learn, int(batchIT[i]), X_test, Y_test)

    # Stochastic gradient descent to find the regression coefficients for different learning rates
    lRateIT = [0.0001, 0.001, 0.01]
    mseLRate = np.zeros((len(lRateIT), epochN))
    for i in range(len(lRateIT)):
        mseLRate[i, :], betas = SGD(epochN, X_train, Y_train, tp, lamb, lRateIT[i], learn, batch, X_test, Y_test)

    # Stochastic gradient descent to find the regression coefficients for different lambda-values (Ridge)
    if tp == 'Ridge':
        lambIT = [0.0001, 0.001, 0.01, 0.1, 1]
        mseLamb = np.zeros((len(lambIT), epochN))
        for i in range(len(lambIT)):
            mseLamb[i, :], betas = SGD(epochN, X_train, Y_train, tp, lambIT[i], lRate, learn, batch, X_test, Y_test)

    # Predict test set
    Y_test_pred = X_test @ betas
    xx_test, yy_test = testParams(n, testSize)

    # Plot figures
    FrankPlot(xx, yy, z, plot=figureInp)
    FrankPlot(xx_test, yy_test, Y_test_pred, plot=figureInp)
    MSESGDSTANDARD(mseStand, plot=figureInp)
    MSESGD(mseBatch, batchIT, "batch size", plot=figureInp)
    MSESGD(mseLRate, lRateIT, "learning rate", plot=figureInp)
    if tp == 'Ridge':
        MSESGD(mseLamb, lambIT, "lambda-values", plot=figureInp)



elif (str(part) == "b" or str(part) == "c" or str(part) == "all"):

    # Exercise b and c)
    z = z.ravel()
    # Input values
    layersN, neuron, hiddenFunc, learn, outputFunc, alpha, tp = inputsssB()

    # Setting up the polynomial design matrix with/without noise
    designMatrix = createDesignmatrix(x, y, poly)
    designMatrix = designMatrix.ravel()
    # Splitting the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(designMatrix, z, test_size=testSize, random_state=40)
    #X_train, X_test, Y_train, Y_test, randomNumber = dataSplit(designMatrix, z, testSize, 0, rn)
    #rn = randomNumber

    # Creating the neural network
    layers = [neuron, layersN]
    if learn == "sigmoid":
        learningRates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    else:
        learningRates = [0.0000001, 0.000001, 0.00001, 0.0001] # 0.001 and higher does not work for RELU
    testMSE = np.zeros(len(learningRates))
    r2 = np.zeros(len(learningRates))
    epochN = 500
    batch = 10

    for i in range(len(learningRates)):
        model = NeuralNetwork(
            X = X_train,
            y = Y_train,
            hiddenNeurons=layers,
            learningRate=learningRates[i],
            batch_size=batch,
            learningType=learn,
            epochsN=epochN,
            activationType=hiddenFunc,
            outputFunctionType=outputFunc,
            alpha=alpha)
        model.train()
        Y_test_pred = model.predict(X_test)
        testMSE[i] = MSE(Y_test_pred, Y_test)
        r2[i] = r2_score(Y_test, Y_test_pred)

    # Sci-kit functionality
    testMSE_sci = np.zeros(len(learningRates))
    r2_sci = np.zeros(len(learningRates))

    for i in range(len(learningRates)):
        modelSci = MLPRegressor(hidden_layer_sizes=neuron,
                                activation=outputFunc,
                                solver='sgd',
                                alpha=alpha,
                                batch_size=batch,
                                learning_rate=learn,
                                learning_rate_init=learningRates[i],
                                max_iter=500,
                                shuffle=True).fit(X_train, Y_train)
        Y_test_pred_Sci= modelSci.predict(X_test)
        testMSE_sci[i] = MSE(Y_test_pred_Sci, Y_test)
        r2_sci[i] = r2_score(Y_test, Y_test_pred_Sci)

    # Printing useful information
    minMSE = np.argmin(testMSE)
    minMSESci = np.argmin(testMSE_sci)
    maxR2 = np.argmax(r2)
    maxR2Sci = np.argmax(r2_sci)
    bestLearnRateMSE = learningRates[minMSE]
    bestLearnRateR2 = learningRates[maxR2]
    bestLearnRateMSESci = learningRates[minMSESci]
    bestLearnRateR2Sci = learningRates[maxR2Sci]
    print("The lowest MSE for the neural network is ", testMSE[minMSE], " and is given by the learning rate ", bestLearnRateMSE)
    print("The highest R2 for the neural network is ", r2[maxR2], " and is given by the learning rate ", bestLearnRateR2)
    print("The lowest MSE for the Scikit neural network is ", testMSE_sci[minMSESci], " and is given by the learning rate ", bestLearnRateMSESci)
    print("The highest R2 for the regression is ", r2_sci[maxR2Sci], " and is given by the learning rate ", bestLearnRateR2Sci)
    learnOpt = bestLearnRateMSE

    # Plot my implementation versus the Scikit implementation
    MSEvsLRATE(testMSE, testMSE_sci, learningRates, plot=figureInp)
    R2vsLRATE(r2, r2_sci, learningRates, plot=figureInp)

    # Now we find the optimal number of layers and neurons
    layersFind = np.arange(1, 20, 1)
    neuronFind = np.arange(10, 201, 10)

    mseFind = np.zeros((len(layersFind), len(neuronFind)))
    mseFindSci = np.zeros((len(layersFind), len(neuronFind)))
    r2Find = np.zeros((len(layersFind), len(neuronFind)))
    r2FindSci = np.zeros((len(layersFind), len(neuronFind)))

    for i in range(len(layersFind)):
        for j in range(len(neuronFind)):
            print("[Number of layers: ", i+1, ", number of neurons: ", j+1, "]")
            # My neural network
            model = NeuralNetwork(
                X=X_train,
                y=Y_train,
                hiddenNeurons=[neuronFind[j], layersFind[i]],
                learningRate=learnOpt,
                batch_size=batch,
                learningType=learn,
                epochsN=epochN,
                activationType=hiddenFunc,
                outputFunctionType=outputFunc,
                alpha=alpha)
            model.train()
            Y_test_pred_find = model.predict(X_test)
            mseFind[i, j] = MSE(Y_test_pred_find, Y_test)
            r2Find[i, j] = R2(Y_test_pred_find, Y_test)

            # Sci kit implementation
            modelSci = MLPRegressor(hidden_layer_sizes=neuronFind[j],
                                    activation=outputFunc,
                                    solver='sgd',
                                    alpha=alpha,
                                    batch_size=batch,
                                    learning_rate=learn,
                                    learning_rate_init=learnOpt,
                                    max_iter=500,
                                    shuffle=True).fit(X_train, Y_train)
            Y_test_pred_Sci = modelSci.predict(X_test)
            mseFindSci[i, j] = MSE(Y_test_pred_Sci, Y_test)
            r2FindSci[i, j] = R2(Y_test, Y_test_pred_Sci)

    # Plotting heatmaps
    # MSE
    heatmap(mseFind, neuronFind, layersFind, 'mse', plot=figureInp)
    heatmap(mseFindSci, neuronFind, layersFind, 'mse', plot=figureInp)
    # R2
    heatmap(r2Find, neuronFind, layersFind, 'r2', plot=figureInp)
    heatmap(r2FindSci, neuronFind, layersFind, 'r2', plot=figureInp)


elif (str(part) == "d" or str(part) == "all"):
    print("Coming")

elif (str(part) == "e" or str(part) == "all"):
    print("Coming")

elif (str(part) == "f" or str(part) == "g" or str(part) == "all"):

    qq = z.ravel()[:, np.newaxis]
    print(qq.shape)

    x1 = x2 = np.linspace(0, 1, n)
    p = int((poly + 1) * (poly + 2) / 2)

    X = np.ones((n, p))

    for i in range(1, poly + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = x1 ** (i - k) * x2 ** k


    X = np.c_[x1.ravel()[:, np.newaxis], x2.ravel()[:, np.newaxis]]

    print(X.shape)