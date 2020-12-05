import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d

def heatPlot(x, ana, fe, diff, t1, t2, t3, L, dx, plot):
    # Plot the heat distribution at given times
    if plot == True:
        ttt = [t1, t2, t3]
        xxx = np.arange(0,L+L/10,L/10)
        for i in range(len(ttt)):
            plt.plot(x, ana[i], label = "Analytical solution at t = " + str(ttt[i]), linewidth=4, color = 'blue')
            plt.plot(x, fe[i], label="Forward Euler solution at t = " + str(ttt[i]), linewidth=4, color='red', linestyle = 'dotted')
            plt.plot(x, diff[i], label="Difference in solutions at t = " + str(ttt[i]), linewidth=4, color='green', linestyle = 'dashed')
            plt.xticks(xxx)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.suptitle('Solution of the heat equation over rod length using dx = ' + str(dx), fontsize=25,
                        fontweight="bold")
            plt.ylabel('u(x,t)', fontsize=20)
            plt.xlabel('Position on rod', fontsize=20)
            plt.legend(loc="upper left", prop={'size': 15})
            plt.grid()
            plt.show()

def heatPlotNN(t, x, dt, dx, L, evaluate, ana, plot):
    # Plot the heat distribution at given times for neural network implementation
    if plot==True:
        # Mesh
        xxx = np.arange(0, L + L / 10, L / 10)
        evaluateT = evaluate.reshape((len(t), len(x))).T
        anaT = ana.reshape((len(t), len(x))).T
        diffT = np.abs(evaluateT-anaT)

        T, X = np.meshgrid(t, x)

        # Index positions
        idx1 = 0
        idx2 = int(int(1/dt)/2)
        idx3 = int(1/dt)

        # Define three different moments in time
        time1 = t[idx1]
        time2 = t[idx2]
        time3 = t[idx3]

        # Take slice of 3d data for the NN at three different times
        nn1 = evaluateT[:, idx1]
        nn2 = evaluateT[:, idx2]
        nn3 = evaluateT[:, idx3]

        # Take slice of 3d data for the analytical solution at three different times
        ana1 = anaT[:, idx1]
        ana2 = anaT[:, idx2]
        ana3 = anaT[:, idx3]

        # Slices of the threeD data at t1, t2 and t3
        plt.figure(figsize=(20, 20))
        plt.plot(x, ana1, label = "Analytical solution at t = " + str(time1), linewidth = 4, color = "blue")
        plt.plot(x, nn1, label = "Neural network solution at t = " + str(time1), linewidth = 4, color = "red", linestyle = 'dotted')
        plt.xticks(xxx)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.suptitle('Solution of the heat equation over rod length using the Tensorflow neural network', fontsize=25,
                     fontweight="bold")
        plt.ylabel('u(x,t)', fontsize=20)
        plt.xlabel('Position on rod', fontsize=20)
        plt.legend(loc="upper left", prop={'size': 15})
        plt.grid()
        #plt.show()

        plt.figure(figsize=(20, 20))
        plt.plot(x, ana2, label = "Analytical solution at t = " + str(time2), linewidth = 4, color = "blue")
        plt.plot(x, nn2, label = "Neural network solution at t = " + str(time2), linewidth = 4, color = "red", linestyle = 'dotted')
        plt.xticks(xxx)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.suptitle('Solution of the heat equation over rod length using the Tensorflow neural network', fontsize=25,
                     fontweight="bold")
        plt.ylabel('u(x,t)', fontsize=20)
        plt.xlabel('Position on rod', fontsize=20)
        plt.legend(loc="upper left", prop={'size': 15})
        plt.grid()
        #plt.show()

        plt.figure(figsize=(20, 20))
        plt.plot(x, ana3, label = "Analytical solution at t = " + str(time3), linewidth = 4, color = "blue")
        plt.plot(x, nn3, label = "Neural network solution at t = " + str(time3), linewidth = 4, color = "red", linestyle = 'dotted')
        plt.xticks(xxx)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.suptitle('Solution of the heat equation over rod length using the Tensorflow neural network', fontsize=25,
                     fontweight="bold")
        plt.ylabel('u(x,t)', fontsize=20)
        plt.xlabel('Position on rod', fontsize=20)
        plt.legend(loc="upper left", prop={'size': 15})
        plt.grid()
        #plt.show()

        plt.show()

def threeD(t, x, dt, dx, evaluate, ana, neurons, layers, plot):
    # Plot 3D heat distribution
    if plot == True:
        evaluateT = evaluate.reshape((len(t), len(x))).T
        anaT = ana.reshape((len(t), len(x))).T
        diffT = np.abs(evaluateT-anaT)

        T, X = np.meshgrid(t, x)

        fig = plt.figure(figsize=(20, 20))
        ax = fig.gca(projection="3d")
        ax.set_title("Deep neural network solution with " + str(layers) + " layers and " + str(neurons) + " neurons")
        f = ax.plot_surface(T, X, evaluateT, linewidth = 0, antialiased = False, cmap=cm.get_cmap("coolwarm"))

        ax.set_xlabel("Time t")
        ax.set_ylabel("Position on rod x")

        fig = plt.figure(figsize=(20, 20))
        ax = fig.gca(projection="3d")
        ax.set_title("Analytical solution to the heat equation")
        f = ax.plot_surface(T, X, anaT, linewidth = 0, antialiased = False, cmap=cm.get_cmap("coolwarm"))
        ax.set_xlabel("Time t")
        ax.set_ylabel("Position on rod x")

        fig = plt.figure(figsize=(20, 20))
        ax = fig.gca(projection="3d")
        ax.set_title("Difference between neural network and analytical solution")
        f = ax.plot_surface(T, X, diffT, linewidth = 0, antialiased = False, cmap=cm.get_cmap("coolwarm"))
        ax.set_xlabel("Time t")
        ax.set_ylabel("Position on rod x")

def eigenPlot(t, v, plot):
    if plot == True:
        plt.figure(figsize=(20, 20))
        plt.plot(t, v, linewidth = 4)
        plt.suptitle('Different eigenvalues over time...', fontsize=25,
                    fontweight="bold")
        plt.legend(["Eigenvector 1", "Eigenvector 2", "Eigenvector 3","Eigenvector 4",
                    "Eigenvector 5","Eigenvector 6"], loc="upper left", prop={'size': 15})
        plt.ylabel('Value', fontsize=20)
        plt.xlabel('Time', fontsize=20)
        plt.grid()
        plt.show()