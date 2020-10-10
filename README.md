# FYS-STK4155
Homework and projects for FYS-STK4155 the fall of 2020

The homework is projects are unrelated. 

# Project 1 

The project one files can be found in the folder "Project1". Here you will find two python files, where the main is the "Project1PF.py" and another file called "FunctionsDef.py" which simply includes functions called by the main file. The python code is written using Pycharm, but the files should run just fine without that particular developer tool. The report can be found under the folder "Report". The report is written using LaTex, so you have to click on the PDF file to view the final report. The terrain data cna be found in the folder "tifdata" and the figures can be found in the folder "Figures". 

As for testing/validation purposes, you can simply run the "Project1PF.py" file as it will ask you for input. Let's say you want to recreate figure 12 and 13 in the report. You would then have to run the file "Project1PF.py" and enter the parameters (100,10,yes,yes,0.001,yes,b,0.25,100). Then the figures will pop up on your screen. The input will always make it clear what it is asking, so it should be straight forward to recreate any result in the report.

In this project I aim to create the basic tools of a linear regression based machine learning algorithm and then utilize these tools in order to predict data the algorithm have not trained on.

Exercises a-c aim to implement the OLS regression scheme as well as the bootstrap and cross-validation resampling methods. Different number of bootstrap iterations and cross-validation fold size are tested and optimal values are found. 

Exercise d and e introduces shrinkage methods to compare to the OLS. The first shrinkage method is the Ridge regression and with it I investigate the effects of the shrinkage parameter as function of model complexity and MSE. The same analysis is repeated for the Lasso regression scheme.

Lastly in exercise f and g, real data is introduced and three models are trained to predict this new data using the 10-fold CV resampling method. One OLS, one Ridge and one Lasso model are fit using the optimal parameters found in the exercise. Then, the results are compared and conclusions about the three algorithms are drawn.
