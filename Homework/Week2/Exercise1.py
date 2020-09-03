import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as skl

def MSE(y_data, y_pred):
    n = np.size(y_data)
    return (np.sum((y_data-y_pred)**2))/n

# Create data
ndata = 1000
x = np.random.rand(ndata)
y = 2+5*x+x+0.1*np.random.rand(ndata)

# Design matrix
poly = 3
X = np.zeros((len(x),poly))
for i in range(0,poly):
    X[:,i] = x**i

# Data split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,train_size=0.75)
standardize = StandardScaler()
standardize.fit(X_train)

# Ordinary least squares
LeastSquares = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
print("Beta coefficients for ordinary least squares: " , LeastSquares)

# Predict
Y_train_pred = X_train @ LeastSquares
Y_test_pred = X_test @ LeastSquares

print("Training MSE for ordinary least squares: " , MSE(Y_train,Y_train_pred), ".")
print("Test MSE for ordinary least squares: " , MSE(Y_test,Y_test_pred), ".")

# Ridge regression
nlammy = 20
lammy = np.logspace(-4,1,nlammy)
Identity = np.eye(poly,poly)
MSE_train_pred_Ridge_sci = np.zeros(nlammy)
MSE_test_pred_Ridge_sci = np.zeros(nlammy)
MSE_train_pred_Lasso_sci = np.zeros(nlammy)
MSE_test_pred_Lasso_sci = np.zeros(nlammy)
MSE_train_pred = np.zeros(nlammy)
MSE_test_pred = np.zeros(nlammy)
beta0 = np.zeros(nlammy)
beta1 = np.zeros(nlammy)
beta2 = np.zeros(nlammy)

for i in range(nlammy):
    Ridge_sci = skl.Ridge(alpha=lammy[i]).fit(X_train, Y_train)
    Y_train_pred_Ridge_sci = Ridge_sci.predict(X_train)
    Y_test_pred_Ridge_sci = Ridge_sci.predict(X_test)

    Lasso_sci = skl.Lasso(alpha=lammy[i]).fit(X_train, Y_train)
    Y_train_pred_Lasso_sci = Lasso_sci.predict(X_train)
    Y_test_pred_Lasso_sci = Lasso_sci.predict(X_test)

    MSE_train_pred_Ridge_sci[i] = MSE(Y_train, Y_train_pred_Ridge_sci)
    MSE_test_pred_Ridge_sci[i] = MSE(Y_test, Y_test_pred_Ridge_sci)

    MSE_train_pred_Lasso_sci[i] = MSE(Y_train, Y_train_pred_Lasso_sci)
    MSE_test_pred_Lasso_sci[i] = MSE(Y_test, Y_test_pred_Lasso_sci)

    # My own ridge (no scikit)
    Ridge = np.linalg.inv(X_train.T @ X_train + lammy[i] * Identity) @ X_train.T @ Y_train
    Y_train_pred_Ridge = X_train @ Ridge
    Y_test_pred_Ridge = X_test @ Ridge

    MSE_train_pred[i] = MSE(Y_train,Y_train_pred_Ridge)
    MSE_test_pred[i] = MSE(Y_test,Y_test_pred_Ridge)

    # Variance calculations
    variance = np.linalg.inv(X_train.T @ X_train + lammy[i] * Identity)
    beta0[i] = variance[0,0]
    beta1[i] = variance[1,1]
    beta2[i] = variance[2,2]


# Plots
plt.figure()
plt.plot(np.log10(lammy),MSE_train_pred,"b--",label = "MSE Ridge Training")
plt.plot(np.log10(lammy),MSE_test_pred,"g",label = "MSE Ridge Test")
plt.plot(np.log10(lammy), MSE_train_pred_Ridge_sci, "c--", label ="MSE Ridge Train (SciKit)")
plt.plot(np.log10(lammy), MSE_test_pred_Ridge_sci, "k", label ="MSE Ridge Test (SciKit)")
plt.plot(np.log10(lammy), MSE_train_pred_Lasso_sci, "m--", label ="MSE Lasso Train (SciKit)")
plt.plot(np.log10(lammy), MSE_test_pred_Lasso_sci, "y", label ="MSE Lasso Test (SciKit)")
plt.xlabel("log10(lambda)")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Variance of optimal lambda
lammyOpt = np.min(MSE_test_pred_Ridge_sci)
print("Variance of beta coefficients using optimal lambda are: \n", "Intercept: ", (np.linalg.inv(X_train.T @ X_train + lammyOpt*Identity)[0,0]), "\n Beta1: ", (np.linalg.inv(X_train.T @ X_train + lammyOpt*Identity)[1,1]), "\n Beta2: ", (np.linalg.inv(X_train.T @ X_train + lammyOpt*Identity)[2,2]))

# Different lambda
plt.figure()
plt.plot(np.log10(lammy),beta0,"b-",label = "beta0")
plt.plot(np.log10(lammy),beta1,"g-.",label = "beta1")
plt.plot(np.log10(lammy),beta2,"r--",label = "beta2")
plt.xlabel("Values of lambda")
plt.ylabel("Variance")
plt.title("Variance of beta coefficients as function of lambda using Ridge regression")
plt.legend()
plt.show()