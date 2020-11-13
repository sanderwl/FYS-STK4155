import numpy as np
from FunctionsDef import inputsss, FrankeFunc, standardize, createDesignmatrix, dataSplit, MSE, R2, gradient, \
    inputsssA, testParams, normalize, SGD, addNoise, inputsssB, FrankeFuncNN, createDesignmatrixNN, getNumbers, \
    findLearnRate, findNeuronLayers, bestModel, learnAlpha, SGDLog
from PlotFunctionsDef import FrankPlot, FrankPlotDiff, MSESGD, MSEvsLRATE, R2vsLRATE, MSESGDSTANDARD, heatmap, \
    plotNumbers, heatmap2, AccvsLRATE, AccvsLRATE2, AccvsLRATEMine, AccvsLRATESci
from NeuralNetworkReg import NeuralNetwork
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier

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
    X_train, X_test, Y_train, Y_test, randomNumber = dataSplit(designMatrix, zSci, testSize, 0, rn)
    rn = randomNumber

    # Stochastic gradient descent to find the regression coefficients using standard parameters
    lRate = 0.01
    batch = int(n / 10)
    mseStand, betas, rr22 = SGD(epochN, X_train, Y_train, tp, lamb, lRate, learn, int(batch), X_test, Y_test)

    # Stochastic gradient descent to find the regression coefficients for different batch sizes
    batchIT = [n, n / 10, n / n]
    mseBatch = np.zeros((len(batchIT), epochN))
    for i in range(len(batchIT)):
        mseBatch[i, :], betas, rr33 = SGD(epochN, X_train, Y_train, tp, lamb, lRate, learn, int(batchIT[i]), X_test, Y_test)

    # Stochastic gradient descent to find the regression coefficients for different learning rates
    lRateIT = [0.0001, 0.001, 0.01]
    mseLRate = np.zeros((len(lRateIT), epochN))
    for i in range(len(lRateIT)):
        mseLRate[i, :], betas, rr44 = SGD(epochN, X_train, Y_train, tp, lamb, lRateIT[i], learn, batch, X_test, Y_test)

    # Stochastic gradient descent to find the regression coefficients for different lambda-values (Ridge)
    if tp == 'Ridge':
        lambIT = [0.0001, 0.001, 0.01, 0.1, 1]
        mseLamb = np.zeros((len(lambIT), epochN))
        for i in range(len(lambIT)):
            mseLamb[i, :], betas, rr55 = SGD(epochN, X_train, Y_train, tp, lambIT[i], lRate, learn, batch, X_test, Y_test)

    # Predict test set
    Y_test_pred = X_test @ betas
    xx_test, yy_test = testParams(n, testSize)

    # Plot figures
    FrankPlot(xx, yy, zSci, plot=figureInp)
    #FrankPlot(xx_test, yy_test, Y_test_pred, plot=figureInp)
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
    designMatrix = createDesignmatrix(x, y, poly)

    # Splitting the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(designMatrix, zSci, test_size=testSize, random_state=40)

    # Creating the neural network
    layers = [neuron, layersN]
    learningRates = [0.00001, 0.0001, 0.001, 0.01]
    epochN = 2000
    batch = 2
    hiddenFunc = ["sigmoid", "relu", "leaky relu"]

    # Declare vectors
    testMSESig = np.zeros(len(learningRates))
    testMSELeakyRelu = np.zeros(len(learningRates))
    testMSERelu = np.zeros(len(learningRates))
    r2Sig = np.zeros(len(learningRates))
    r2LeakyRelu = np.zeros(len(learningRates))
    r2Relu = np.zeros(len(learningRates))
    testMSE_sci_Relu = np.zeros(len(learningRates))
    r2_sci_Relu = np.zeros(len(learningRates))

    # Train and fit neural network to the data
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

    mseFindSig = np.zeros((len(layersFind), len(neuronFind)))
    r2FindSig = np.zeros((len(layersFind), len(neuronFind)))
    mseFindRelu = np.zeros((len(layersFind), len(neuronFind)))
    r2FindRelu = np.zeros((len(layersFind), len(neuronFind)))
    mseFindSciRelu = np.zeros((len(layersFind), len(neuronFind)))
    r2FindSciRelu = np.zeros((len(layersFind), len(neuronFind)))
    mseFindLeakyRelu = np.zeros((len(layersFind), len(neuronFind)))
    r2FindLeakyRelu = np.zeros((len(layersFind), len(neuronFind)))

    # Find optimal number of hidden neurons and layers
    for k in range(len(hiddenFunc)):
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
                    activationType=hiddenFunc[k],
                    outputFunctionType=outputFunc,
                    alpha=alpha)
                model.train()
                Y_test_pred_find = model.predict(X_test)

                check = np.isnan(Y_test_pred_find)
                if hiddenFunc[k] == "sigmoid":
                    mseFindSig[i, j] = MSE(Y_test_pred_find, Y_test)
                    if True in check:
                        r2FindSig[i] = None
                    else:
                        r2FindSig[i] = r2_score(Y_test, Y_test_pred_find)
                elif hiddenFunc[k] == "relu":
                    mseFindRelu[i, j] = MSE(Y_test_pred_find, Y_test)
                    if True in check:
                        r2FindRelu[i] = None
                    else:
                        r2FindRelu[i] = r2_score(Y_test, Y_test_pred_find)
                elif hiddenFunc[k] == "leaky relu":
                    mseFindLeakyRelu[i, j] = MSE(Y_test_pred_find, Y_test)
                    if True in check:
                        r2FindLeakyRelu[i] = None
                    else:
                        r2FindLeakyRelu[i] = r2_score(Y_test, Y_test_pred_find)

    # Sci kit implementation
    for i in range(len(layersFind)):
        for j in range(len(neuronFind)):
            modelSci = MLPRegressor(hidden_layer_sizes=neuronFind[j],
                                    activation="relu",
                                    solver='sgd',
                                    alpha=alpha,
                                    batch_size=batch,
                                    learning_rate=learn,
                                    learning_rate_init=learnOptSig,
                                    max_iter=epochN,
                                    shuffle=True).fit(X_train, Y_train)
            Y_test_pred_Sci = modelSci.predict(X_test)
            mseFindSciRelu[i, j] = MSE(Y_test_pred_Sci, Y_test)
            check2 = np.isnan(Y_test_pred_Sci)
            if True in check2:
                r2FindSciRelu[i, j] = None
            else:
                r2FindSciRelu[i, j] = r2_score(Y_test, Y_test_pred_Sci)

    # Plotting heatmaps
    # MSE
    heatmap(mseFindSig, neuronFind, layersFind, 'mse', "sigmoid", plot=figureInp)
    heatmap(mseFindRelu, neuronFind, layersFind, 'mse', "relu", plot=figureInp)
    heatmap(mseFindSciRelu, neuronFind, layersFind, 'mse', "relu", plot=figureInp)
    heatmap(mseFindLeakyRelu, neuronFind, layersFind, 'mse', "leaky relu", plot=figureInp)
    # R2
    heatmap(r2FindSig, neuronFind, layersFind, 'r2', "sigmoid", plot=figureInp)
    heatmap(r2FindRelu, neuronFind, layersFind, 'r2', "relu", plot=figureInp)
    heatmap(r2FindSciRelu, neuronFind, layersFind, 'r2', "relu", plot=figureInp)
    heatmap(r2FindLeakyRelu, neuronFind, layersFind, 'r2', "leaky relu", plot=figureInp)


elif (str(part) == "d" or str(part) == "all"):

    # Read the digit data
    num_train_images = 1000
    num_test_images = 1000
    X_train, Y_train, X_test, Y_test = getNumbers(num_train_images, num_test_images)
    # plotNumbers(numbers, figureInp)

    # Standardize the data
    X_train = standardize(X_train, scaleInp)
    X_test = standardize(X_test, scaleInp)

    # Defining network parameters
    layers = [20,20,20]
    epochN = 1000
    batch = 10
    hiddenFunc = ["sigmoid", "relu"]
    hiddenFunc2 = ["logistic", "relu"]
    outputFunc = "softmax"

    # We need the 1hot format
    Y_train = Y_train.reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto')
    Y_train_1hot = encoder.fit_transform(Y_train).toarray()
    Y_test_1hot = encoder.fit_transform(Y_test.reshape(-1, 1)).toarray()

    # Flatten the image
    n_inputs = len(X_train)
    n_inputs2 = len(X_test)
    inputs = X_train.reshape(n_inputs, -1)
    inputs2 = X_test.reshape(n_inputs2, -1)

    # More parameters
    learningRates = [0.0001, 0.001, 0.01, 0.1]
    alphas = [0.0001, 0.001, 0.01, 0.1]

    # Find the optimal learning rate
    OptimalLearningRateOLS, AccSig, AccRelu, AccSciSig, AccSciRelu = findLearnRate(learningRates, alphas, hiddenFunc,
                                                                                       hiddenFunc2, inputs, Y_train_1hot,
                                                                                       inputs2, Y_test, layers,epochN,
                                                                                       batch)
    print("Optimal learning rate for sigmoid: ", OptimalLearningRateOLS[0])
    print("Optimal learning rate for RELU: ", OptimalLearningRateOLS[1])
    AccvsLRATE(AccSig, AccRelu, AccSciSig, AccSciRelu, learningRates, figureInp)

    # Finding optimal number of neurons and layers
    layersFind = np.arange(1, 10, 1)
    neuronFind = np.arange(1, 101, 10)
    optTotal, AccFindSig, AccFindRelu, AccFindSciSig, AccFindSciRelu = findNeuronLayers(layersFind, neuronFind,
                                                                                        hiddenFunc, hiddenFunc2, inputs,
                                                                                        Y_train_1hot, inputs2, Y_test,
                                                                                        layers, OptimalLearningRateOLS,
                                                                                        epochN, batch, [0, 0])

    print("Optimal number of neurons/layers for sigmoid: ", optTotal[0])
    print("Optimal number of neurons/layers for RELU: ", optTotal[1])

    # Best model
    #AccSig_epic, AccRelu_epic, AccSciSig_epic, AccSciRelu_epic = bestModel(optTotal, hiddenFunc, hiddenFunc2, inputs,
    #                                                                       Y_train_1hot, inputs2, Y_test, layers,
    #                                                                       OptimalLearningRateOLS, epochN, batch, [0,0])
    #print("Best accuracy for sigmoid model: ", AccSig_epic)
    #print("Best accuracy for RELU model: ", AccRelu_epic)

    # Penalty parameter for Ridge
    OptimalLearningRate_Ridge, OptimalAlpha_Ridge, AccSig_Ridge, AccRelu_Ridge, AccSciSig_Ridge, AccSciRelu_Ridge = learnAlpha(
                                                                                                           learningRates,
                                                                                                           alphas,
                                                                                                           hiddenFunc,
                                                                                                           hiddenFunc2,
                                                                                                           inputs,
                                                                                                           Y_train_1hot,
                                                                                                           inputs2,
                                                                                                           Y_test,
                                                                                                           layers,
                                                                                                           epochN,
                                                                                                           batch)

    print("Optimal learning rate for Ridge using sigmoid: ", OptimalLearningRate_Ridge[0])
    print("Optimal learning rate for Ridge using RELU: ", OptimalLearningRate_Ridge[1])
    print("Optimal penalty for Ridge using sigmoid: ", OptimalAlpha_Ridge[0])
    print("Optimal penalty for Ridge using RELU: ", OptimalAlpha_Ridge[1])

    # Optimal number of hidden neurons and layers for Ridge
    optTotal_Ridge, AccFindSig_Ridge, AccFindRelu_Ridge, AccFindSciSig_Ridge, AccFindSciRelu_Ridge = findNeuronLayers(
                                                                                        layersFind,
                                                                                        neuronFind,
                                                                                        hiddenFunc, hiddenFunc2, inputs,
                                                                                        Y_train_1hot, inputs2, Y_test,
                                                                                        layers, OptimalLearningRateOLS,
                                                                                        epochN, batch, OptimalAlpha_Ridge)

    print("Optimal number of neurons/layers for Ridge using sigmoid: ", optTotal_Ridge[0])
    print("Optimal number of neurons/layers for Ridge using RELU: ", optTotal_Ridge[1])

    # Best model for Ridge
    #AccSig_epic_Ridge, AccRelu_epic_Ridge, AccSciSig_epic_Ridge, AccSciRelu_epic_Ridge = bestModel(optTotal, hiddenFunc,
     #                                                                                              hiddenFunc2, inputs,
     #                                                                                              Y_train_1hot, inputs2,
     #                                                                                              Y_test, layers,
     #                                                                                              OptimalLearningRateOLS,
     #                                                                                              epochN, batch, OptimalAlpha_Ridge)
    #print("Best accuracy for Ridge sigmoid model: ", AccSig_epic_Ridge)
    #print("Best accuracy for Ridge RELU model: ", AccRelu_epic_Ridge)


    print("----------------------------")
    print("Optimal learning rate for sigmoid: ", OptimalLearningRateOLS[0])
    print("Optimal learning rate for RELU: ", OptimalLearningRateOLS[1])

    print("Optimal number of neurons/layers for sigmoid: ", optTotal[0])
    print("Optimal number of neurons/layers for RELU: ", optTotal[1])

    #print("Best accuracy for sigmoid model: ", AccSig_epic)
    #print("Best accuracy for RELU model: ", AccRelu_epic)

    print("Optimal learning rate for Ridge using sigmoid: ", OptimalLearningRate_Ridge[0])
    print("Optimal learning rate for Ridge using RELU: ", OptimalLearningRate_Ridge[1])
    print("Optimal penalty for Ridge using sigmoid: ", OptimalAlpha_Ridge[0])
    print("Optimal penalty for Ridge using RELU: ", OptimalAlpha_Ridge[1])

    print("Optimal number of neurons/layers for Ridge using sigmoid: ", optTotal_Ridge[0])
    print("Optimal number of neurons/layers for Ridge using RELU: ", optTotal_Ridge[1])

    #print("Best accuracy for Ridge sigmoid model: ", AccSig_epic_Ridge)
    #print("Best accuracy for Ridge RELU model: ", AccRelu_epic_Ridge)


    # Plot accuracy vs learning rate
    AccvsLRATE(AccSig, AccRelu, AccSciSig, AccSciRelu, learningRates, figureInp)

    # Plot heatmap of different layer and neuron sizes
    heatmap(AccFindSig, neuronFind, layersFind, 'Accuracy', "sigmoid", plot=figureInp)
    heatmap(AccFindRelu, neuronFind, layersFind, 'Accuracy', "relu", plot=figureInp)
    heatmap(AccFindSciSig, neuronFind, layersFind, 'Accuracy', "sigmoid", plot=figureInp)
    heatmap(AccFindSciRelu, neuronFind, layersFind, 'Accuracy', "relu", plot=figureInp)

    # Plot heatmap of different layer and neuron sizes Ridge
    heatmap(AccFindSig_Ridge, neuronFind, layersFind, 'Accuracy', "sigmoid", plot=figureInp)
    heatmap(AccFindRelu_Ridge, neuronFind, layersFind, 'Accuracy', "relu", plot=figureInp)
    heatmap(AccFindSciSig_Ridge, neuronFind, layersFind, 'Accuracy', "sigmoid", plot=figureInp)
    heatmap(AccFindSciRelu_Ridge, neuronFind, layersFind, 'Accuracy', "relu", plot=figureInp)

    # Plot heatmap of learning rate and penalty for Ridge
    heatmap2(AccSig_Ridge, learningRates, alphas, plot=figureInp)
    heatmap2(AccRelu_Ridge, learningRates, alphas, plot=figureInp)
    heatmap2(AccSciSig_Ridge, learningRates, alphas, plot=figureInp)
    heatmap2(AccSciRelu_Ridge, learningRates, alphas, plot=figureInp)

elif (str(part) == "e" or str(part) == "all"):

    # Read the digit data
    num_train_images = 1000
    num_test_images = 1000
    X_train, Y_train, X_test, Y_test = getNumbers(num_train_images, num_test_images)
    # plotNumbers(numbers, figureInp)

    # Standardize the data
    X_train = standardize(X_train, scaleInp)
    X_test = standardize(X_test, scaleInp)

    # Defining SGD parameters
    epochN = 1000
    batch = 10

    # We need the 1hot format
    Y_train = Y_train.reshape(-1, 1)
    encoder = OneHotEncoder(categories='auto')
    Y_train_1hot = encoder.fit_transform(Y_train).toarray()
    Y_test_1hot = encoder.fit_transform(Y_test.reshape(-1, 1)).toarray()

    # Flatten the image
    n_inputs = len(X_train)
    n_inputs2 = len(X_test)
    inputs = X_train.reshape(n_inputs, -1)
    inputs2 = X_test.reshape(n_inputs2, -1)

    # More parameters
    learningRates = [0.0001, 0.001, 0.01, 0.1]
    alphas = [0.0001, 0.001, 0.01, 0.1]

    # Declare vectors
    accuracyOLS = np.zeros(len(learningRates))
    accuracyOLSSci = np.zeros(len(learningRates))
    accuracyRidge = np.zeros((len(alphas), len(learningRates)))
    accuracyRidgeSci = np.zeros((len(alphas), len(learningRates)))

    # Logistic SGD for OLS
    for i in range(len(learningRates)):
        print(i)
        accuracyOLS[i] = SGDLog(epochN, inputs, Y_train_1hot, 'OLS', 0, learningRates[i], 'constant', batch, inputs2, Y_test_1hot)
        SGDOLSSci = SGDClassifier(alpha=0, learning_rate='constant', eta0=learningRates[i], random_state=69)
        SGDOLSSci.fit(inputs, np.argmax(Y_test_1hot, axis=1))
        accuracyOLSSci[i] = SGDOLSSci.score(inputs, np.argmax(Y_test_1hot, axis=1))

    # Logistic SGD for Ridge
    for i in range(len(alphas)):
        print(i)
        for j in range(len(learningRates)):
            accuracyRidge[i, j] = SGDLog(epochN, inputs, Y_train_1hot, 'Ridge', alphas[i], learningRates[j], 'constant', batch, inputs2, Y_test_1hot)
            SGDRidgeSci = SGDClassifier(alpha=alphas[i], learning_rate='constant', eta0=learningRates[j], random_state=69)
            SGDRidgeSci.fit(inputs, np.argmax(Y_test_1hot, axis=1))
            accuracyRidgeSci[i, j] = SGDRidgeSci.score(inputs, np.argmax(Y_test_1hot, axis=1))

    # Plot accuracy vs. learning rate
    AccvsLRATE2(accuracyOLS, accuracyRidge, accuracyOLSSci, accuracyRidgeSci, alphas, learningRates, plot=figureInp)
    AccvsLRATEMine(accuracyOLS, accuracyRidge, alphas, learningRates, plot=figureInp)
    AccvsLRATESci(accuracyOLSSci, accuracyRidgeSci, alphas, learningRates, plot=figureInp)
    heatmap2(accuracyRidge, learningRates, alphas, plot=figureInp)
    heatmap2(accuracyRidgeSci, learningRates, alphas, plot=figureInp)