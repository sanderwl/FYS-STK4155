import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def inputs():
    # Inputs for the total exercise
    print("Intro text.......")
    ex = input("Which exercise should be ran? (string, a/b/c/d/e/all): ")
    figure = input("Plot figures? (yes/no): ")
    if figure == "yes":
        figureInp = True
    else:
        figureInp = False
    own = input("Use predefined parameters or should they be specified? (pre/self): ")
    return "a", True, "pre" #str(ex), bool(figureInp), str(own)

def inputsAB(own):
    # Input parameters for exercise a, but also used in other exercises
    if own == "self":
        dt = input("Step size in time for forward Euler? (float, between 0 and 1): ")
        dx = input("Step size in space for forward Euler? (float, between 0 and 1): ")
        T = input("For what amount of time should the rod be observed? (float, typically a few seconds)")
        L = input("What is the length of the rod? (float, equal to 1 in the exercise): ")
        alpha = input("What is the value of the heat constant alpha? (integer, typically set to 1): ")
    else:
        dt = 0.1
        dx = [1/10, 1/100]
        T = 1
        L = 1
        alpha = 1
    return float(dt), list(dx), float(L), float(T), int(alpha)

def inputsC(own):
    # Inputs for exercise c, but also used in other exercises
    if own == "self":
        layers = input("How many hidden layers in the neural network? (integer): ")
        neurons = input("How many hidden neurons in each layer? (integer): ")
        lrate = input("What is the learning rate? (float, typically between zero and one): ")
        its = input("Number of iterations/epochs to run? (integer, typically very large): ")
    else:
        layers = 10
        neurons = 10
        lrate = 0.001
        its = 100000
    return int(layers), int(neurons), float(lrate), int(its)

def stabilize(alpha, dt, dx):
    # Check if Euler is stable or not and change dt if unstable
    if alpha * dt / (dx ** 2) > 0.5:
        print("Unstable due to alpha*dt/dx^2 > 0.5. Changing parameter dt to stabilize...")
        dt = (alpha * 0.5 * dx**2)
        print("dt have been changed to " + str(dt) + " and the criteria alpha*dt/dx^2 <= 0.5 is now satisfied")
    return dt

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

def makeTens(dt, dx, L, T):
    # Format t and x to the preferred Tensorflow format
    t = np.arange(0, T+dt, dt)
    x = np.arange(0, L+dx, dx)

    X, T = np.meshgrid(x, t)

    tnew = T.ravel().reshape(-1, 1)
    xnew = X.ravel().reshape(-1, 1)

    tTens = tf.convert_to_tensor(tnew)
    xTens = tf.convert_to_tensor(xnew)

    grid = tf.concat([xTens, tTens], 1)

    return t, x, tnew, xnew, tTens, xTens, grid

def temp(x):
    # Initial condition/heat
    t = tf.sin(np.pi*x)
    return t

def tfInit(tTens, xTens, grid, structure, lrate):
    # Set up TF neural network with different neurons and layers
    lastLayer = grid
    for i in range(len(structure)):
        layer = tf.layers.dense(lastLayer, structure[i], activation=tf.nn.sigmoid)
        lastLayer = layer
    output = tf.layers.dense(lastLayer, 1)

    # Define trial function and its gradients
    trial = (1 - tTens) * temp(xTens) + xTens * (1 - xTens) * tTens * output
    dtTrial = tf.gradients(trial, tTens)
    dxTrial = tf.gradients(tf.gradients(trial, xTens), xTens)

    # Define the cost function (MSE)
    error = tf.square(dtTrial[0] - dxTrial[0])
    cost = tf.reduce_mean(error)

    # Define the Adam optimization algorithm
    AdaOpt = tf.train.AdamOptimizer(lrate)
    minAdaOpt = AdaOpt.minimize(cost)

    return trial, minAdaOpt

def tfTrainEval(tnew, xnew, its, minAdaOpt, trial):
    # Gather and initialize all defined variables
    evaluate = None
    initialize = tf.global_variables_initializer()

    # Train and evaluate the neural network
    with tf.Session() as session:

        #print("The initial cost is: %g"%cost.eval())
        initialize.run()
        for i in range(its):
            session.run(minAdaOpt)
        #print("The cost after training: %g"%cost.eval())

        evaluate = trial.eval()

    # Compare the neural network results to the analytical result
    ana = np.sin(xnew*np.pi)*np.exp(tnew*-np.pi**2)
    diff = np.abs(evaluate-ana)

    return evaluate, ana, diff












