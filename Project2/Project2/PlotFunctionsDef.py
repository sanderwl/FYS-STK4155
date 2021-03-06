import numpy as np
from numpy import diff
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def FrankPlot(x,y,z,plot):
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

def FrankPlotDiff(x,y,z,z2,plot):
    if plot == True:
        # Plot the Franke function
        zz = z-z2
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, zz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # Labels
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Color map grid
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

def MSESGDSTANDARD(mseThing, plot):
    if plot == True:
        # Plot the MSE as function of epoch iterations
        x = np.arange(0, len(mseThing), 1)
        xxx = np.arange(0,len(mseThing),len(mseThing)/10)
        plt.plot(x, mseThing, label="MSE", linewidth=4)
        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.suptitle("MSE as function of epoch iterations using standard parameters", fontsize=25, fontweight="bold")
        plt.ylabel('MSE', fontsize=20)
        plt.xlabel('Iterations', fontsize=20)
        plt.legend(loc="upper right", prop={'size': 20})
        plt.show()

def MSESGD(mseThing, f, st, plot):
    if plot == True:
        # Plot the MSE as function of epoch iterations with an additional parameter (learning rate or batch size)
        x = np.arange(0, len(mseThing[0]), 1)
        xxx = np.arange(0,len(mseThing[0]),len(mseThing[0])/10)
        for i in range(len(mseThing)):
            plt.plot(x, mseThing[i, :], label=str(st) + " = " + str(f[i]), linewidth=4)
        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.suptitle("MSE as function of epoch iterations for different values of the " + str(st), fontsize=25, fontweight="bold")
        plt.ylabel('MSE', fontsize=20)
        plt.xlabel('Iterations', fontsize=20)
        plt.legend(loc="upper right", prop={'size': 20})
        plt.show()

def MSEvsLRATE(testMSESig, testMSERelu, testMSE_SciRelu, testMSELeakyRelu, learningRates, plot):
    if plot == True:
        # Plot the MSE versus the learning rate
        xxx = np.log10(learningRates)
        plt.plot(xxx, testMSESig, label="Sigmoid (NN)", linewidth=4)
        plt.plot(xxx, testMSERelu, label="Relu (NN)", linewidth=4)
        plt.plot(xxx, testMSE_SciRelu, ':', label="Relu (Scikit)", linewidth=4)
        plt.plot(xxx, testMSELeakyRelu, label="Leaky Relu (NN)", linewidth=4)
        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.suptitle('MSE comparison of different implementations', fontsize=25,
                     fontweight="bold")
        plt.ylabel('MSE', fontsize=20)
        plt.xlabel('Logarithm of learning rates', fontsize=20)
        plt.legend(loc="upper right", prop={'size': 20})
        plt.grid()
        plt.show()

def R2vsLRATE(R2Sig, R2Relu, R2_sciRelu, R2LeakyRelu, learningRates, plot):
    if plot == True:
        # Plot the R2 versus the learning rate
        xxx = np.log10(learningRates)
        plt.plot(xxx, R2Sig, label="Sigmoid (NN)", linewidth=4)
        plt.plot(xxx, R2Relu, label="Relu (NN)", linewidth=4)
        plt.plot(xxx, R2_sciRelu, ':', label="Relu (Scikit)", linewidth=4)
        plt.plot(xxx, R2LeakyRelu, label="Leaky Relu (NN)", linewidth=4)
        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.suptitle('R2 comparison of different implementations', fontsize=25,
                     fontweight="bold")
        plt.ylabel('R2', fontsize=20)
        plt.xlabel('Logarithm of learning rates', fontsize=20)
        plt.legend(loc="lower right", prop={'size': 20})
        plt.grid()
        plt.show()

def AccvsLRATE(Acc1, Acc2, Acc3, Acc4, learningRates, plot):
    if plot == True:
        # Plot the accuracy as function of learning rate
        xxx = np.log10(learningRates)
        plt.plot(xxx, Acc1, label="My NN implementation (sigmoid)", linewidth=4)
        plt.plot(xxx, Acc2, label="My NN implementation (RELU)", linewidth=4)
        plt.plot(xxx, Acc3, label="Scikit NN implementation (sigmoid)", linewidth=4)
        plt.plot(xxx, Acc4, label="Scikit NN implementation (RELU)", linewidth=4)
        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.suptitle('Accuracy comparison of different implementations', fontsize=25,
                     fontweight="bold")
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('Logarithm of learning rates', fontsize=20)
        plt.legend(loc="lower right", prop={'size': 20})
        plt.grid()
        plt.show()

def AccvsLRATE2(AccOLS, AccRidge, AccOLSSci, AccRidgeSci, alphas, learningRates, plot):
    if plot == True:
        # Plot the accuracy as function of learning rate
        xxx = np.log10(learningRates)
        plt.plot(xxx, AccOLS, label="SGD accuracy for OLS", linewidth=4, color = 'blue')
        plt.plot(xxx, AccOLSSci, label="SGD accuracy for OLS (Scikit)", linewidth=4, color = 'blue', linestyle = 'dashed')
        colars = ['red', 'green', 'purple', 'black']
        for i in range(len(AccRidge)):
            plt.plot(xxx, AccRidge[i,:], label="SGD accuracy for Ridge with penalty " + str(alphas[i]), linewidth=4, color= colars[i])
            plt.plot(xxx, AccRidgeSci[i, :], label="SGD accuracy for Ridge (Scikit) with penalty " + str(alphas[i]),
                     linewidth=4, linestyle = 'dashed', color= colars[i])

        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.suptitle('Accuracy comparison of different implementations', fontsize=25,
                     fontweight="bold")
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('Logarithm of learning rates', fontsize=20)
        plt.legend(loc="lower left", prop={'size': 15})
        plt.grid()
        plt.show()

def AccvsLRATEMine(AccOLS, AccRidge, alphas, learningRates, plot):
    if plot == True:
        # Plot the accuracy as function of learning rate
        xxx = np.log10(learningRates)
        plt.plot(xxx, AccOLS, label="SGD accuracy for OLS", linewidth=4, color = 'blue')
        colars = ['red', 'green', 'purple', 'black']
        for i in range(len(AccRidge)):
            plt.plot(xxx, AccRidge[i,:], label="SGD accuracy for Ridge with penalty " + str(alphas[i]), linewidth=4, color= colars[i])

        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.suptitle('Accuracy comparison of different implementations', fontsize=25,
                     fontweight="bold")
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('Logarithm of learning rates', fontsize=20)
        plt.legend(loc="lower left", prop={'size': 15})
        plt.grid()
        plt.show()

def AccvsLRATESci(AccOLSSci, AccRidgeSci, alphas, learningRates, plot):
    if plot == True:
        # Plot the accuracy as function of learning rate
        xxx = np.log10(learningRates)
        plt.plot(xxx, AccOLSSci, label="SGD accuracy for OLS (Scikit)", linewidth=4, color = 'blue', linestyle = 'dashed')
        colars = ['red', 'green', 'purple', 'black']
        for i in range(len(AccRidgeSci)):
            plt.plot(xxx, AccRidgeSci[i, :], label="SGD accuracy for Ridge (Scikit) with penalty " + str(alphas[i]),
                     linewidth=4, linestyle = 'dashed', color= colars[i])

        plt.xticks(xxx)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.suptitle('Accuracy comparison of different implementations', fontsize=25,
                     fontweight="bold")
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('Logarithm of learning rates', fontsize=20)
        plt.legend(loc="lower left", prop={'size': 15})
        plt.grid()
        plt.show()

def heatmap(x, neurons, layers, rm, mains, plot):
    if plot == True:
        # Plot heatmap for different values of hidden neurons and layers
        plt.figure()
        ax = sns.heatmap(x, xticklabels=neurons, yticklabels=layers, annot=True)
        plt.suptitle(rm + ' comparison of number of layers and neurons using the ' + mains + " activation function"
                     , fontsize=25, fontweight="bold")
        ax.set_xlabel('Number of hidden neurons')
        ax.set_ylabel('Number of hidden layers')
        plt.show()

def heatmap2(x, learningRate, Penalty, plot):
    if plot == True:
        # Plot heatmap for different values of learning rate and penalty
        plt.figure()
        ax = sns.heatmap(x, xticklabels=learningRate, yticklabels=Penalty, annot=True)
        plt.suptitle('Accuracy comparison different values of learning rate and penalty parameters'
                     , fontsize=25, fontweight="bold")
        ax.set_xlabel('Learning rates')
        ax.set_ylabel('Penalty')
        plt.show()

def plotNumbers(data, plot):
    if plot == True:
        # Plot the digit data (example)
        image = np.asarray(data[7]).squeeze()
        plt.imshow(image)
        plt.show()