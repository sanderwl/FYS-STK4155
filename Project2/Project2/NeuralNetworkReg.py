import numpy as np


class NeuralNetwork:
    def __init__(
            self,
            X,
            y,
            hiddenNeurons = [10],
            epochsN = 1000,
            batch_size = 10,
            learningRate = 0.01,
            learningType = 'constant',
            initialBias = 0.01,
            activationType = 'sigmoid',
            outputFunctionType = 'identity',
            alpha = 0):

        self.X_train = X
        self.Y_train = y

        self.inputsN = X.shape[0]
        self.featuresN = X.shape[1]
        self.categoriesN = y.shape[1]

        self.hiddenNeurons = hiddenNeurons
        self.layers = ([self.featuresN] + self.hiddenNeurons + [self.categoriesN])
        self.layersN = len(self.layers)

        self.batch_size = batch_size
        self.batchesN = int(self.inputsN / self.batch_size)
        self.epochsN = epochsN
        self.learningType = learningType
        self.learningRate = learningRate / batch_size

        self.activationType = activationType
        self.activationFunction(self.activationType)
        self.outputFunctionType = outputFunctionType
        self.outputFunction(self.outputFunctionType)

        self.weights = [None]  # weights for every layer
        self.initialBias = initialBias # bias for every layer
        self.biases = [None]
        self.alpha = alpha

        self.a = [None]  # neuron output
        self.z = [None]  # activation function
        self.backError = [None]  # error for back propagation
        self.costs = np.zeros(self.epochsN) # cost function

        for l in range(1, self.layersN):
            self.weights.append(np.random.normal(loc=0.0,scale=np.sqrt(2 / (self.layers[l - 1] + self.layers[l])), size=(self.layers[l - 1], self.layers[l])))
            self.biases.append(np.zeros(self.layers[l]) + self.initialBias)
            self.z.append(None)
            self.a.append(None)
            self.backError.append(None)

    def feedForward(self):
        self.a[0] = self.X
        for l in range(1, self.layersN):
            self.z[l] = self.a[l - 1] @ self.weights[l] + self.biases[l]
            self.a[l] = self.act_func(self.z[l])

        # Last layers of neurons is the output layer
        self.a[-1] = self.output_func(self.z[-1])

    def feedForwardOut(self, X):
        a = X
        for l in range(1, self.layersN):
            z = a @ self.weights[l] + self.biases[l]
            a = self.act_func(z)

        # Last layers of neurons is the output layer
        a = self.output_func(z)
        return a

    def backPropagation(self):

        self.cost = self.mse(self.a[-1])

        self.backError[-1] = self.a[-1] - self.y

        self.backErrorW = self.a[-2].T @ self.backError[-1] + self.alpha * self.weights[-1]
        self.backErrorB = np.sum(self.backError[-1], axis=0)

        self.weights[-1] -= self.learningRate * self.backErrorW
        self.biases[-1] -= self.learningRate * self.backErrorB

        for l in range(self.layersN - 2, 0, -1):
            self.backError[l] = (self.backError[l + 1] @ self.weights[l + 1].T * self.act_func_der(self.z[l]))

            self.backErrorW = self.a[l - 1].T @ self.backError[l] + self.alpha * self.weights[l]

            self.weights[l] -= self.learningRate * self.backErrorW
            self.biases[l] -= self.learningRate * np.sum(self.backError[l], axis=0)

    def train(self):

        for i in range(self.epochsN):
            dexxer = 0
            idx = np.arange(self.inputsN)
            np.random.shuffle(idx)
            for batch in range(self.batchesN):

                self.schedule(i * self.batchesN + batch)

                rand_indeces = idx[dexxer * self.batch_size:(dexxer + 1) * self.batch_size]
                self.X = self.X_train[rand_indeces, :]
                self.y = self.Y_train[rand_indeces]

                self.feedForward()
                self.backPropagation()

                dexxer += 1

            self.costs[i] = self.cost

    def predict_class(self, X):
        output = self.feedForwardOut(X)
        return np.argmax(output, axis=1)

    def predict(self, X):
        return self.feedForwardOut(X)

    def schedule(self, t):
        if self.learningType == 'adaptive':
            t0 = 5
            t1 = 50
            self.learningRate = 0.01 * t0 / (t + t1)
            print(self.learningRate)
        elif self.learningType == 'constant':
            self.learningRate = self.learningRate

    def activationFunction(self, act_func_str):
        if act_func_str == 'sigmoid':
            self.act_func = self.sigmoid
            self.act_func_der = self.sigmoidDerivative
        elif act_func_str == 'tanh':
            self.act_func = self.tanh
            self.act_func_der = self.tanhDerivative
        elif act_func_str == 'relu':
            self.act_func = self.relu
            self.act_func_der = self.reluDerivative
        elif act_func_str == 'leaky relu':
            self.act_func = self.leakyRelu
            self.act_func_der = self.leakyReluDerivative

    def outputFunction(self, output_func_str='softmax'):

        if output_func_str == 'sigmoid':
            self.output_func = self.sigmoid
        if output_func_str == 'tanh':
            self.output_func = self.tanh
        elif output_func_str == 'relu':
            self.output_func = self.relu
            self.output_func_der = self.reluDerivative
        elif output_func_str == 'softmax':
            self.output_func = self.softmax
        elif self.outputFunctionType == 'identity':
            self.output_func = self.linear

    def mse(self, x):
        return 0.5 * ((x - self.y) ** 2).mean()

    def mseDerivative(self, x):
        return x - self.y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanhDerivative(self, x):
        return 1 - self.tanh(x) ** 2

    def softmax(self, x):
        exp_term = np.exp(x)
        return exp_term / exp_term.sum(axis=1, keepdims=True)

    def relu(self, x):
        #check = np.isnan(x)
        #if True in check:
        #    print("Nan in relu")
        return np.maximum(x, 0)

    def reluDerivative(self, x):
        return 1 * (x >= 0)

    def leakyRelu(self, x):
        smallV = 0.01
        return np.maximum(x, smallV)

    def leakyReluDerivative(self, x):
        return 1 * (x >= 0)

    def linear(self, x):
        return x