import numpy as np
from HeatEquation import analytical, forwardEuler
from FunctionDef import inputs, inputsAB, inputsC, stabilize
from FunctionDefPlot import heatPlot, heatPlotNN, threeD, slices
from NeuralNetwork import deep_neural_network, solve_pde_deep_neural_network, g_analytic, g_trial

# Inputs for all parts of the project
ex, figureInp = inputs()

if (ex == "a" or ex == "b" or ex == "all"):
    # Inputs for exercise a and b
    dt, dx, L, alpha = inputsAB()

    for i in range(len(dx)):
        # Check is algorithm is stable for forward Euler and change dt
        dt = 0.1
        dt = stabilize(alpha, dt, dx[i])

        # Define spatial units
        gridp = int(L / dx[i])
        x = np.linspace(0, L, gridp + 1)

        # Define temporal units
        t1 = 0
        t2 = 0.5
        timep = int(t2 / dt)

        # Analytical solution to the heat equation
        ana = analytical(x, alpha, t2)

        # Forward Euler solution to the heat equation
        fe = forwardEuler(x, timep, gridp, dt, dx[i])

        # Find difference between analytical and forward Euler solution
        diff = ana-fe

        # Plot heat curve at for t2 for previously defined
        heatPlot(x, ana, fe, diff, t2, L, dx[i], plot=figureInp)

elif (ex == "c" or ex == "all"):

    # Inputs for exercise c
    dt, dx, L, alpha, own = inputsAB()
    layers, neurons = inputsC(own)

    # Stabilize
    dt = stabilize(alpha, dt, dx)
    t2 = 0.5

    timestep = int(1 / dt)
    spacestep = int(1 / dx)
    t = np.linspace(0, 1, timestep)
    x = np.linspace(0, L, spacestep)

    its = 100
    lrate = 0.1

    structure = np.repeat(neurons, layers)

    # Neural network implementation
    P = solve_pde_deep_neural_network(x, t, structure, its, lrate)

    ## Store the results
    g_dnn_ag = np.zeros((spacestep, timestep))
    G_analytical = np.zeros((spacestep, timestep))

    for i,x_ in enumerate(x):
        print(100*(i+1)/x, " percent complete xD.")
        for j, t_ in enumerate(t):
            point = np.array([x_, t_])
            g_dnn_ag[i, j] = g_trial(point, P)

            G_analytical[i, j] = g_analytic(point)

    # Find the map difference between the analytical and the computed solution
    diff_ag = np.abs(g_dnn_ag - G_analytical)
    print('Max absolute difference between the analytical solution and the network: %g'%np.max(diff_ag))

    # plot
    threeD(t, x, structure, g_dnn_ag, G_analytical, diff_ag, figureInp)
    slices(t, x, timestep, g_dnn_ag, G_analytical, figureInp)







