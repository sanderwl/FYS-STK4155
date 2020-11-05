import numpy as np
from FunctionsDef import inputsss, FrankeFunc, standardize, createDesignmatrix, dataSplit, MSE, R2, gradient, \
    inputsssA, testParams, normalize, SGD, addNoise, inputsssB, FrankeFuncNN, createDesignmatrixNN
from PlotFunctionsDef import FrankPlot, FrankPlotDiff, MSESGD, MSEvsLRATE, R2vsLRATE, MSESGDSTANDARD, heatmap
from NeuralNetworkReg import NeuralNetwork
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
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
zz = FrankeFuncNN(xx, yy)
zz = standardize(zz, scalee=scaleInp)
zz = normalize(zz, scalee=scaleInp)
zz = addNoise(zz, noiseLevel=noiseLVL)
z = zz

zzSci = FrankeFunc(xx, yy)
zzSci = standardize(zzSci, scalee=scaleInp)
zzSci = normalize(zzSci, scalee=scaleInp)
zzSci = addNoise(zzSci, noiseLevel=noiseLVL)
zSci = zzSci

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

    # Input values
    layersN, neuron, hiddenFuncInp, learn, outputFunc, alpha, tp = inputsssB()

    # Setting up the polynomial design matrix with/without noise
    designMatrix = createDesignmatrixNN(x, y, poly)
    designMatrixSci = createDesignmatrix(x, y, poly)

    # Splitting the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(designMatrixSci, zSci, test_size=testSize, random_state=40)

    # Creating the neural network
    layers = [neuron, layersN]
    learningRates = [0.00001, 0.0001, 0.001, 0.01]
    epochN = 100
    batch = 10
    hiddenFunc = ["sigmoid", "relu", "leaky relu"]

    testMSESig = np.zeros(len(learningRates))
    testMSELeakyRelu = np.zeros(len(learningRates))
    testMSERelu = np.zeros(len(learningRates))
    r2Sig = np.zeros(len(learningRates))
    r2LeakyRelu = np.zeros(len(learningRates))
    r2Relu = np.zeros(len(learningRates))
    testMSE_sci_Relu = np.zeros(len(learningRates))
    r2_sci_Relu = np.zeros(len(learningRates))

    for k in range(len(hiddenFunc)):
        for i in range(len(learningRates)):
            print("lrate: ", learningRates[i])
            model = NeuralNetwork(
                X = X_train,
                y = Y_train,
                hiddenNeurons=layers,
                learningRate=learningRates[i],
                batch_size=batch,
                learningType=learn,
                epochsN=epochN,
                activationType=hiddenFunc[k],
                outputFunctionType=outputFunc,
                alpha=alpha)
            model.train()
            Y_test_pred = model.predict(X_test)
            if hiddenFunc[k] == "sigmoid":
                testMSESig[i] = MSE(Y_test_pred, Y_test)
                checks4 = np.isnan(Y_test_pred)
                if True in checks4:
                    r2Sig[i] = None
                else:
                    r2Sig[i] = r2_score(Y_test, Y_test_pred)
            elif hiddenFunc[k] == "relu":
                testMSERelu[i] = MSE(Y_test_pred, Y_test)
                checks5 = np.isnan(Y_test_pred)
                if True in checks5:
                    r2Relu[i] = None
                else:
                    r2Relu[i] = r2_score(Y_test, Y_test_pred)
            elif hiddenFunc[k] == "leaky relu":
                testMSELeakyRelu[i] = MSE(Y_test_pred, Y_test)
                checks6 = np.isnan(Y_test_pred)
                if True in checks6:
                    r2LeakyRelu[i] = None
                else:
                    r2LeakyRelu[i] = r2_score(Y_test, Y_test_pred)

    # Sci-kit functionality for relu
    for i in range(len(learningRates)):
        modelSci = MLPRegressor(hidden_layer_sizes=neuron,
                                activation="relu",
                                solver='sgd',
                                alpha=alpha,
                                batch_size=batch,
                                learning_rate=learn,
                                learning_rate_init=learningRates[i],
                                max_iter=epochN,
                                shuffle=True).fit(X_train, Y_train)
        Y_test_pred_Sci= modelSci.predict(X_test)
        checks = np.isnan(Y_test_pred_Sci)
        if True in checks:
            r2_sci_Relu[i] = None
        else:
            r2_sci_Relu[i] = r2_score(Y_test, Y_test_pred_Sci)
        testMSE_sci_Relu[i] = MSE(Y_test_pred_Sci, Y_test)

    # Printing useful information
    minMSESig = np.argmin(testMSESig)
    maxR2Sig = np.argmax(r2Sig)
    bestLearnRateMSESig = learningRates[minMSESig]
    bestLearnRateR2Sig = learningRates[maxR2Sig]
    print("The lowest MSE for my neural network is (Sigmoid) ", testMSESig[minMSESig], " and is given by the learning rate ", bestLearnRateMSESig)
    print("The highest R2 for my neural network is (Sigmoid) ", r2Sig[maxR2Sig], " and is given by the learning rate ", bestLearnRateR2Sig)
    learnOptSig = bestLearnRateMSESig
    print("Optimal learning rate using the Sigmoid activation function: ", learnOptSig)
    print("----------------------------------------")

    minMSERelu = np.argmin(testMSERelu)
    minMSESciRelu = np.argmin(testMSE_sci_Relu)
    maxR2Relu = np.argmax(r2Relu)
    maxR2SciRelu = np.argmax(r2_sci_Relu)
    bestLearnRateMSERelu = learningRates[minMSERelu]
    bestLearnRateR2Relu = learningRates[maxR2Relu]
    bestLearnRateMSESciRelu = learningRates[minMSESciRelu]
    bestLearnRateR2SciRelu = learningRates[maxR2SciRelu]
    print("The lowest MSE for my neural network is (Relu) ", testMSERelu[minMSERelu], " and is given by the learning rate ", bestLearnRateMSERelu)
    print("The highest R2 for my neural network is (Relu) ", r2Relu[maxR2Relu], " and is given by the learning rate ", bestLearnRateR2Relu)
    print("The lowest MSE for the Scikit neural network is (Relu) ", testMSE_sci_Relu[minMSESciRelu], " and is given by the learning rate ", bestLearnRateMSESciRelu)
    print("The highest R2 for the Scikit neural network is (Relu) ", r2_sci_Relu[maxR2SciRelu], " and is given by the learning rate ", bestLearnRateR2SciRelu)
    learnOptRelu = bestLearnRateMSERelu
    print("Optimal learning rate using the Relu activation function: ", learnOptRelu)
    print("----------------------------------------")

    minMSELeakyRelu = np.argmin(testMSELeakyRelu)
    maxR2LeakyRelu = np.argmax(r2LeakyRelu)
    bestLearnRateMSELeakyRelu = learningRates[minMSELeakyRelu]
    bestLearnRateR2LeakyRelu = learningRates[maxR2LeakyRelu]
    print("The lowest MSE for my neural network is (Leaky Relu) ", testMSELeakyRelu[minMSELeakyRelu], " and is given by the learning rate ", bestLearnRateMSELeakyRelu)
    print("The highest R2 for my neural network is (Leaky Relu) ", r2LeakyRelu[maxR2LeakyRelu], " and is given by the learning rate ", bestLearnRateR2LeakyRelu)
    learnOptLeakyRelu = bestLearnRateMSELeakyRelu
    print("Optimal learning rate using the Leaky Relu activation function: ", learnOptLeakyRelu)

    # Plot my implementation versus the Scikit implementation
    MSEvsLRATE(testMSESig, testMSERelu, testMSE_sci_Relu, testMSELeakyRelu, learningRates, plot=figureInp)
    R2vsLRATE(r2Sig, r2Relu, r2_sci_Relu, r2LeakyRelu, learningRates, plot=figureInp)

    # Now we find the optimal number of layers and neurons
    layersFind = np.arange(1, 10, 1)
    neuronFind = np.arange(1, 101, 10)

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
                learningRate=learnOptSig,
                batch_size=batch,
                learningType=learn,
                epochsN=epochN,
                activationType=hiddenFuncInp,
                outputFunctionType=outputFunc,
                alpha=alpha)
            model.train()
            Y_test_pred_find = model.predict(X_test)
            mseFind[i, j] = MSE(Y_test_pred_find, Y_test)
            check = np.isnan(Y_test_pred_find)
            if True in check:
                r2Find[i, j] = 0
            else:
                r2Find[i, j] = r2_score(Y_test, Y_test_pred_find)

            # Sci kit implementation
            modelSci = MLPRegressor(hidden_layer_sizes=neuronFind[j],
                                    activation="relu",
                                    solver='sgd',
                                    alpha=alpha,
                                    batch_size=batch,
                                    learning_rate=learn,
                                    learning_rate_init=learnOptSig,
                                    max_iter=500,
                                    shuffle=True).fit(X_train, Y_train)
            Y_test_pred_Sci = modelSci.predict(X_test)
            mseFindSci[i, j] = MSE(Y_test_pred_Sci, Y_test)
            check2 = np.isnan(Y_test_pred_find)
            if True in check2:
                r2FindSci[i, j] = 0
            else:
                r2FindSci[i, j] = r2_score(Y_test, Y_test_pred_Sci)

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

    print(np.max(np.random.randn(10, 5)))