import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def heatPlot(x, ana, fe, diff, t2, L, dx, plot):
    if plot == True:
        xxx = np.arange(0,L+L/10,L/10)
        plt.plot(x, ana, label = "Analytical solution at t = " + str(t2), linewidth=4, color = 'blue')
        plt.plot(x, fe, label="Forward Euler solution at t = " + str(t2), linewidth=4, color='red', linestyle = 'dotted')
        plt.plot(x, diff, label="Difference in solutions at t = " + str(t2), linewidth=4, color='green', linestyle = 'dashed')
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

def heatPlotNN(x, error, errorana, plot):
    if plot == True:
        print("xd")

def threeD(t, x, structure, g_dnn_ag, G_analytical, diff_ag, plot):
    if plot == True:
        T, X = np.meshgrid(t, x)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.set_title('Solution from the deep neural network w/ %d layer' % len(structure))
        s = ax.plot_surface(T, X, g_dnn_ag, linewidth=0, antialiased=False, cmap=cm.viridis)
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Position $x$');

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.set_title('Analytical solution')
        s = ax.plot_surface(T, X, G_analytical, linewidth=0, antialiased=False, cmap=cm.viridis)
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Position $x$');

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')
        ax.set_title('Difference')
        s = ax.plot_surface(T, X, diff_ag, linewidth=0, antialiased=False, cmap=cm.viridis)
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Position $x$');

def slices(t, x, timestep, g_dnn_ag, G_analytical, plot):
    if plot==True:
        indx1 = 0
        indx2 = int(timestep / 2)
        indx3 = timestep - 1

        t1 = t[indx1]
        t2 = t[indx2]
        t3 = t[indx3]

        # Slice the results from the DNN
        res1 = g_dnn_ag[:, indx1]
        res2 = g_dnn_ag[:, indx2]
        res3 = g_dnn_ag[:, indx3]

        # Slice the analytical results
        res_analytical1 = G_analytical[:, indx1]
        res_analytical2 = G_analytical[:, indx2]
        res_analytical3 = G_analytical[:, indx3]

        # Plot the slices
        plt.figure(figsize=(10, 10))
        plt.title("Computed solutions at time = %g" % t1)
        plt.plot(x, res1)
        plt.plot(x, res_analytical1)
        plt.legend(['dnn', 'analytical'])

        plt.figure(figsize=(10, 10))
        plt.title("Computed solutions at time = %g" % t2)
        plt.plot(x, res2)
        plt.plot(x, res_analytical2)
        plt.legend(['dnn', 'analytical'])

        plt.figure(figsize=(10, 10))
        plt.title("Computed solutions at time = %g" % t3)
        plt.plot(x, res3)
        plt.plot(x, res_analytical3)
        plt.legend(['dnn', 'analytical'])

        plt.show()