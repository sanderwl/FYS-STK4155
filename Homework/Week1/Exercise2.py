import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = np.random.rand(10,1)
y = 2.0+5*x*x+0.1*np.random.randn(10,1)

linreg = LinearRegression()
linreg.fit(x,y)
ypredict = linreg.predict(x)


plt.plot(x,ypredict,"r-")
plt.plot(x,y,'ro')
#plt.axis([0,2,0,15])
plt.xlabel('x')
plt.ylabel(r'$y$')
plt.title(r"random numbers")
plt.show()

print(mean_squared_error(y, ypredict))
