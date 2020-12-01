import numpy as np

def analytical(x, alpha, t2):
    # Return the analytical solution to the heat equation
    ana = np.sin(np.pi * x) * np.exp(-np.pi**2 * alpha**2 * t2)
    return ana

def forwardEuler(x, timep, gridp, dt, dx):

    # Initial condition
    initial = np.sin(x*np.pi)
    fe = initial

    # Propagate heat using forward Euler
    for i in range(timep+1):
        fe[1:gridp] = fe[1:gridp] + dt/(dx**2)*(fe[2:]-2*fe[1:gridp] + fe[0:gridp-1])


    return fe