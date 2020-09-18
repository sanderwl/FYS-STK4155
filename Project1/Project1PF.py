import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from FunctionsDef import FrankePlot, MSE, R2,StdPolyOLS

# Create data
n = 10
x = np.arange(0, 1, 1/n)
y = np.arange(0, 1, 1/n)
xx, yy = np.meshgrid(x,y)

#z = FrankePlot(xx, yy)

#poly = 5
#OLSOLS = StdPolyOLS(poly,x,y)


N = len(x)
l = int((n+1)*(n+2)/2)
X = np.ones((N,l))

for i in range(1,n+1):
    q = int((i)*(i+1)/2)
    print(q)
        for k in range(i+1):
            print(k)
	        X[:,q+k] = (x**(i-k))*(y**k)

    


