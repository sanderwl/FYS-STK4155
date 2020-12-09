import numpy as np
from numpy import linalg
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from FunctionDef import inputs, inputsAB, inputsC, stabilize, makeTens, temp, tfInit, tfTrainEval, analytical, \
    forwardEuler, tfEigen, makeTensEigen, eigen, inputsD
from FunctionDefPlot import heatPlot, heatPlotNN, threeD, eigenPlot
from NeuralNetwork import deep_neural_network, solve_pde_deep_neural_network, g_analytic, g_trial

# Inputs for all parts of the project
ex, figureInp, own = inputs()

if (ex == "a" or ex == "b" or ex == "all"):
    # Inputs for exercise a and b
    dt, dx, L, T, alpha = inputsAB(own)
    dtt = dt
    # Run forward Euler for both values of dx
    for i in range(len(dx)):
        # Check is algorithm is stable for forward Euler and change dt
        dt = dtt
        dt = stabilize(alpha, dt, dx[i])

        # Define spatial units
        gridp = int(L / dx[i])
        x = np.linspace(0, L, gridp + 1)

        # Define temporal units
        t1 = 0
        t2 = 0.5
        t3 = 1
        timep1 = int(t1 / dt)
        timep2 = int(t2 / dt)
        timep3 = int(t3 / dt)

        # Analytical solutions to the heat equation
        ana1 = analytical(x, alpha, t1)
        ana2 = analytical(x, alpha, t2)
        ana3 = analytical(x, alpha, t3)

        # Forward Euler solutions to the heat equation
        fe1 = forwardEuler(x, timep1, gridp, dt, dx[i])
        fe2 = forwardEuler(x, timep2, gridp, dt, dx[i])
        fe3 = forwardEuler(x, timep3, gridp, dt, dx[i])

        # Find difference between analytical and forward Euler solutions
        diff1 = ana1 - fe1
        diff2 = ana2 - fe2
        diff3 = ana3 - fe3

        # Plot heat curve at for t=c
        ana = [ana1, ana2, ana3]
        fe = [fe1, fe2, fe3]
        diff = [diff1, diff2, diff3]
        heatPlot(x, ana, fe, diff, t1, t2, t3, L, dx[i], plot=figureInp)

elif (ex == "c" or ex == "all"):

    # Inputs for exercise c
    dt, dx, L, T, alpha = inputsAB(own)
    dx = dx[0]
    layers, neurons, lrate, its = inputsC(own)
    structure = np.repeat(neurons, layers)

    # Reformat t and x to the preferred Tensorflow format
    t, x, tnew, xnew, tTens, xTens, grid = makeTens(dt, dx, L, T)

    # Initialization/setup the Tensorflow network
    trial, minAdaOpt = tfInit(tTens, xTens, grid, structure, lrate)

    # Training and testing
    evaluate, ana2, diff2 = tfTrainEval(tnew, xnew, its, minAdaOpt, trial)

    # Plotting 3D heat over time and space as well as slices from that 3D plot
    threeD(t, x, dt, dx, evaluate, ana2, neurons, layers, plot=figureInp)
    heatPlotNN(t, x, dt, dx, L, evaluate, ana2, plot=figureInp)

elif (ex == "d" or ex == "all"):
    np.random.seed(7)
    tf.set_random_seed(7)
    # Inputs for exercise d
    dt, dx, L, TT, alpha = inputsAB(own)
    dx = dx[0]
    layers, neurons, lrate, its = inputsC(own)
    precision, k, n = inputsD(own)

    Q = np.random.rand(n, n)
    A = (Q.T + Q) / 2
    A_temp = k * A
    A_tf = tf.convert_to_tensor(A_temp, dtype=tf.float64)

    vv = np.random.rand(n)
    structure = np.repeat(neurons, layers)
    print(A)

    # Numpy eigenvector/values
    if k == -1:
        npEigenVal, npEigenVec = linalg.eig(A)
        eigenValEX = np.max(npEigenVal)
        idx = np.argmax(npEigenVal)
        eigenVecEX = npEigenVec[idx]
    elif k == 1:
        npEigenVal, npEigenVec = linalg.eig(A)
        eigenValEX = np.min(npEigenVal)
        idx = np.argmin(npEigenVal)
        eigenVecEX = npEigenVec[idx]

    t, x, tnew, xnew, vnew, tTens, xTens, vTens, grid, timepoints = makeTensEigen(dt, dx, vv, L, TT, n)

    eigenVec, t = tfEigen(t, tTens, xTens, vTens, grid, structure, lrate, k, n, A, timepoints, its, precision)

    eigenvalues = eigen(eigenVec[-1], A).eval(session=tf.Session())

    print("Smallest numpy eigenvector: ", eigenVecEX)
    print("Smallest NN eigenvector: ", eigenVec[-1])
    print("Smallest NN eigenvector (normalized): ", eigenVec[-1] / linalg.norm(eigenVec[-1]))
    print("Smallest numpy eigenvalue: ", eigenValEX)
    print("Smallest NN eigenvalue: ", eigenvalues)

    print("x", npEigenVal)
    print("y", npEigenVec)

    # Plot
    eigenPlot(t, eigenVec, plot=figureInp)





