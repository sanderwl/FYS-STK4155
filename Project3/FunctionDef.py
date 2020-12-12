import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def inputs():
    # Inputs for the total exercise
    print("This is the code for Project 3 in FYS-STK4155. Exercise a and b finds the forward Euler (and analytical)"
          " solution to the heat equation. Exercise c finds the solution to the heat equation using \n a neural"
          "network code. Exercise d attemps to find the smallest/largest eigenvalue of a 6x6 matrix. Please give"
          "the inputs for the code to run...")
    ex = input("Which exercise should be ran? (string, a/b/c/d/e/all): ")
    figure = input("Plot figures? (yes/no): ")
    if figure == "yes":
        figureInp = True
    else:
        figureInp = False
    own = input("Use predefined parameters or should they be specified? (pre/self): ")
    return str(ex), bool(figureInp), str(own)

def inputsAB(own):
    # Input parameters for exercise a, but also used in other exercises
    if own == "self":
        dt = input("Step size in time for forward Euler? (float, between 0 and 1): ")
        dx = list(map(float,input("Step size in space for forward Euler? (list of floats of size 2, components between 0 and 1):  ").strip().split()))[:2]
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
        layers = 3
        neurons = 10
        lrate = 0.001
        its = 1000000
    return int(layers), int(neurons), float(lrate), int(its)

def inputsD(own):
    # Inputs for exercise c, but also used in other exercises
    if own == "self":
        precision = input("Precision of the eigenvector/value solver? (float, typically 0.0001): ")
        k = input("Find the largest or smallest eigenvector of the matrix? (integer, -1 (smallest) or 1 (largest)): ")
        n = input("Size of the matrix? (integer): ")
        TT = 5
    else:

        precision = 0.00001
        k = -1
        n = 6
        TT = 5
    return float(precision), int(k), int(n), int(TT)

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

        initialize.run()
        for i in range(its):
            session.run(minAdaOpt)

        evaluate = trial.eval()

    # Compare the neural network results to the analytical result
    ana = np.sin(xnew*np.pi)*np.exp(tnew*-np.pi**2)
    diff = np.abs(evaluate-ana)

    return evaluate, ana, diff

def makeTensEigen(dt, dx, vv, L, TT, n):
    # Format t and x to the tensorflow format
    times = int(TT / dt)
    t = np.linspace(0, (times - 1) * dt, times)
    x = np.linspace(1, n, n)

    X, T = np.meshgrid(x, t)
    V, Tf = np.meshgrid(vv, t)

    xnew = (X.ravel()).reshape(-1, 1)
    tnew = (T.ravel()).reshape(-1, 1)
    vnew = (V.ravel()).reshape(-1, 1)

    xTens = tf.convert_to_tensor(xnew, dtype=tf.float64)
    tTens = tf.convert_to_tensor(tnew, dtype=tf.float64)
    vTens = tf.convert_to_tensor(vnew, dtype=tf.float64)

    # Declare the grid that goes into the neural network
    grid = tf.concat([xTens, tTens], 1)

    return t, x, tnew, xnew, vnew, tTens, xTens, vTens, grid, times

def tfEigen(t, tTens, xTens, vTens, grid, structure, lrate, k, n, A, times, its, precision):
    # Declare the dense structure of the neural network using TF
    lastLayer = grid
    for i in range(len(structure)):
        layer = tf.layers.dense(lastLayer, structure[i], activation=tf.nn.sigmoid)
        lastLayer = layer
    output = tf.layers.dense(lastLayer, 1)

    # Declare the trial function and its gradient according to Yi et.al.
    trial = output * tTens + vTens * k
    dtTrial = tf.gradients(trial, tTens)
    trialR = tf.reshape(trial, (times, n))
    dtTrialR = tf.reshape(dtTrial, (times, n))

    # Implement Yi et.al. cost function
    loss0 = 0
    for i in range(times):
        trialF = tf.reshape(trialR[i], (n, 1))
        dtTrialF = tf.reshape(dtTrialR[i], (n, 1))
        ads = YiSol(trialF, A) - trialF
        error = tf.square(-dtTrialF + ads)
        loss0 += tf.reduce_sum(error)

    loss = tf.reduce_sum(loss0 / (n * times), name='loss')

    # Define the Adam optimizer for training
    AdaOpt = tf.train.AdamOptimizer(lrate)
    minAdaOpt = AdaOpt.minimize(loss)

    # Gather and initialize all variables
    initialize = tf.global_variables_initializer()

    # Run TF session for the Adam optimizer
    with tf.Session() as sess:
        # Initialize the variables
        initialize.run()

        print("Initial loss at ", loss.eval())

        for i in range(its):
            if (i % 1000 == 0):
                print("Step ", i, " with loss ", loss.eval())

            sess.run(minAdaOpt)

            # Stop the session if the loss is small enough
            if loss.eval() < precision:
                break

        print("Final loss of ", loss.eval())

        eigenvectors = tf.reshape(trial, (times, n)).eval()

    return eigenvectors, t


def YiSol(x, Mat):
    # This is the Yi et.al. f(x(t)) function
    identity = tf.eye(int(Mat.shape[0]), dtype=tf.float64)
    xTrans = tf.transpose(x)
    value1 = tf.matmul(xTrans, x) * Mat
    value2 = (1 - tf.matmul(tf.matmul(xTrans, Mat), x)) * identity
    value = tf.matmul((value1 + value2), x)
    return value

def eigen(v, Mat):
    # Find eigenvalues from the eigenvector
    v = v.reshape(Mat.shape[0], 1)
    eig1 = tf.matmul(tf.matmul(tf.transpose(v), Mat), v)[0, 0]
    eig2 = tf.matmul(tf.transpose(v), v)[0, 0]
    eig = eig1/eig2
    return eig








