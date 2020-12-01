import numpy as np

def inputs():
    # Inputs for the total exercise
    print("This exercise utilizes the forward Euler method to propagate heat through a rod of length L using "
          "a step size in time dt and in space dx")
    ex = input("Which exercise should be ran? (string, a/b/c/d/e/all): ")
    figure = input("Plot figures? (yes/no): ")
    if figure == "yes":
        figureInp = True
    else:
        figureInp = False
    return "c", True #str(ex), bool(figureInp)

def inputsAB():
    # Input parameters for exercise a
    own = input("Use predefined parameters or should they be specified? (pre/self): ")
    if own == "self":
        dt = input("Step size in time for forward Euler? (float, between 0 and 1): ")
        dx = input("Step size in space for forward Euler? (float, between 0 and 1): ")
        L = input("What is the length of the rod? (integer, equal to 1 in the exercise): ")
        alpha = input("What is the value of the heat constant alpha? (integer, typically set to 1): ")
    else:
        dt = 0.1
        dx = [1/10, 1/100]
        L = 1
        alpha = 1
    return 0.1, 1/10, 1, 1, "pre" #float(dt), list(dx), int(L), int(alpha), str(own)

def inputsC(own):
    # Inputs for exercise c
    if own == "self":
        layers = input("How many hidden layers in the neural network? (integer): ")
        neurons = input("How many hidden neurons in each layer? (integer): ")
    else:
        layers = [10]
        neurons = [10]
    return 2, 2 #list(layers), list(neurons)

def stabilize(alpha, dt, dx):
    # Check if Euler is stable or not
    if alpha * dt / (dx ** 2) > 0.5:
        print("Unstable due to alpha*dt/dx^2 > 0.5. Changing parameter dt to stabilize...")
        dt = (alpha * 0.5 * dx**2)
        print("dt have been changed to " + str(dt) + " and the criteria alpha*dt/dx^2 <= 0.5 is now satisfied")
    return dt















