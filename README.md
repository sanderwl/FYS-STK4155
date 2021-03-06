# FYS-STK4155
Homework and projects for FYS-STK4155 in the fall of 2020

# Project 1 

The project one files can be found in the folder "Project1". Here you will find two python files, where the main is the "Project1PF.py" and another file called "FunctionsDef.py" which simply includes functions called by the main file. The python code is written using Pycharm, but the files should run just fine without that particular developer tool. The report can be found under the folder "Report". The report is written using LaTex, so you have to click on the PDF file to view the final report. The terrain data can be found in the folder "tifdata" and the figures can be found in the folder "Figures". 

As for testing/validation purposes, you can simply run the "Project1PF.py'' file as it will ask you for input. Let's say you want to recreate figure 12 and 13 in the report. You would then have to run the file "Project1PF.py'' and enter the parameters (100,10,yes,yes,0.001,yes,b,0.25,100). Then the figures will pop up on your screen. The input will always make it clear what it is asking, so it should be straightforward to recreate any result in the report.

In this project I aim to create the basic tools of a linear regression based machine learning algorithm and then utilize these tools in order to predict data the algorithm has not trained on.

Exercises a-c aim to implement the OLS regression scheme as well as the bootstrap and cross-validation resampling methods. Different numbers of bootstrap iterations and cross-validation fold size are tested and optimal values are found. 

Exercise d and e introduces shrinkage methods to compare to the OLS. The first shrinkage method is the Ridge regression and with it I investigate the effects of the shrinkage parameter as a function of model complexity and MSE. The same analysis is repeated for the Lasso regression scheme.

Lastly in exercise f and g, real data is introduced and three models are trained to predict this new data using the 10-fold CV resampling method. One OLS, one Ridge and one Lasso model are fit using the optimal parameters found in the exercise. Then, the results are compared and conclusions about the three algorithms are drawn.

# Project 2

The entire project 2 is found in the folder Project2/Project2. This folder contains all the python files used in this exercise and the main file you should run is the "Project2PF.py" (main file). The other files are support files and includes different useful function where the file "FunctionsDef.py" perform various calculations, the file "PlotFunctionsDef.py" plots everything seen in the report and the file "NeuralNetworkReg.py" contains the neural network used in this project. There are two empty files named "NeuralNetworkClassification.py" and "NeuralNetworkClassification2.py" which were simply used for experimentation throughout the making of this project. The IDE used in this project is called Pycharm, but the python files should be able to run for other IDEs/no IDE as well. The report itself can be found under Project2/Project2/Report as both a tex file and a pdf file (pdf file recommended for viewing). The folder Project2/Project2/Report/Data contains the MNIST digit data used in the project and the folder Project2/Project2/Report/Figures contain the figures used in the report. 

The code is interactive so running the file "Project2PF.py'' will allow you to recreate any result in the report. Let's say you want to recreate figure 2 in the report. You would then have to run the file "Project2PF.py'' and enter the parameters (100,5,yes,yes,0.001,yes,a,constant,100,OLS). Then the figures will then pop up on your screen. The input will always make it clear what it is asking, so it should be straightforward to recreate any result in the report.

This project aims to implement a regression and a logistic SGD and neural network algorithms that predict the Franke function (regression case) or the MNIST digit data (classification case). Here, we first consider the regression case in exercise 1a), b) and c) before we move on to the classification case in exercise 1d) and 1e). Lastly, the results are discussed and compared in exercise 1f) and some conclusions are drawn. Several changes in parameters and function are used throughout the report and this variation is discussed in detail. 

# Project 3

Everything related to project 3 can be found in the folder “Project 3”. This includes all the Python code used in the project with the main file being “Project3PF.py”. The other Python files are support files to the main file where “FunctionDef.py” performs most of the calculations and the file “FunctionDefPlot.py” takes care of the plotting. 
An IDE named Pycharm was used in this project, but all the code should be able to run without this IDE.

The report can be found under “Project3/Report/Project3.pdf” (tex file also available) where the figures from the project can be found in the same directory, but under “Project3/Report/Figures”.

The code is interactive meaning that running “Project3PF.py” will recreate any result found in the report. Say you want to recreate figure 4 in the report. You would then have to run “Project3PF.py” and enter the inputs (a, yes, pre). Here, the input “pre” will use the predefined parameters. If you wish to run with your own parameters simply type “self” instead of “pre” and the code will ask you for the input parameters. The input will always be clear in what it is asking, so it should be straightforward to give any output you want.

This project aims to solve the one dimensional heat equation both analytically and numerically. The numerical part includes both a forward Euler scheme and a neural network implementation. The solutions from the numerical implementations are then compared to the analytical solution (exercise a through c). 
Then, the same neural network code is utilized to find the smallest/largest eigenvalue of a real, symmetric and square matrix (exercise d). 
Lastly, the results are discussed and conclusions are drawn together with an evaluation of the methods used.


