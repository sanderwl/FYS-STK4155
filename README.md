# FYS-STK4155
Homework and projects for FYS-STK4155 the fall of 2020

The homework is projects are unrelated. 

# Project 1 

In this project I aim to create the basic tools of a linear regression based machine learning algorithm and then utilize these tools in order to predict data the algorithm have not trained on.

Exercises a-c aim to implement the OLS regression scheme as well as the bootstrap and cross-validation resampling methods. Different number of bootstrap iterations and cross-validation fold size are tested and optimal values are found. 

Exercise d and e introduces shrinkage methods to compare to the OLS. The first shrinkage method is the Ridge regression and with it I investigate the effects of the shrinkage parameter as function of model complexity and MSE. The same analysis is repeated for the Lasso regression scheme.

Lastly in exercise f and g, real data is introduced and three models are trained to predict this new data using the 10-fold CV resampling method. One OLS, one Ridge and one Lasso model are fit using the optimal parameters found in the exercise. Then, the results are compared and conclusions about the three algorithms are drawn.