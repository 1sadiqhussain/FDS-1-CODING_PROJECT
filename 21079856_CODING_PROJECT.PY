import pandas as pd # Imports pandas library as pd
import numpy as np # Imports numpy library as np
from sklearn.linear_model import LinearRegression # Imports Linear Regression library from sklearn.linear_model
import matplotlib.pyplot as plt # Imports matplotlib.pyplot library as plt

# Reading the data into Python numpy array
inputdata6_array = np.array(pd.read_csv("inputdata6.csv")) # Reads the csv file into numpy array
print("Printing the inputdata6 as an array: \n",inputdata6_array) # Prints the data as an array
rainfall = inputdata6_array[:, 0] # Creating the columns as numpy array variables
productivity = inputdata6_array[:, 1] # Creating the columns as numpy array variables
print("The rainfall array column: \n", rainfall) # Printing the Rainfall array column
print("The productivity array column: \n", productivity) # Printing the Rainfall array column

# Creating a Linear Regression model based on the data
mypattern = LinearRegression() # Creating a linear regression model based on the data 
X = rainfall[:, np.newaxis] # Re-shaping rainfall column to a matrix for prediction
print(mypattern.fit(X, productivity)) # To affirm the variable mypattern is a linear regression fit
y_predictions = mypattern.predict(X) # Predicting the productivity using re-shaped X (rainfall)
print("The predicted values of y from linear regression model is: \n", y_predictions) # Printing the Y-Predictions
interception = mypattern.intercept_ # Creating interception coefficient
slope = mypattern.coef_ # Creating slope coefficient

# Linear equation y = a + bx
# Where a is intercept, b is coefficient/slope and x is 245mm
x = 245 # Defining the x value
y_pred = interception + (slope*x) # Caluculating the y_pred 
print("The productivity coefficient of the field if the amount of precipitation is 245mm: ", y_pred) # Printing the productivity coefficient
x = "Rainfall : " + str(245) + "mm" # Defining the x value to print on the plot
y = "Predicted productivity coefficient : " + str(y_pred) # Defining the y value to print on the plot

# Plotting the inputdata6 as a two-dimensional scatter plot
plt.figure(figsize=(8,6)) # To create a figure
plt.scatter(rainfall, productivity, color = 'red', marker='o', edgecolors='black',s=42,label='Rainfall Vs productivity') # Plotting the scatter plot
plt.title('Rainfall vs Productivity', fontsize = 20,color='black') # Defining the title of the plot
plt.xlabel('Rainfall', fontsize = 15) # Defining the X-Label 
plt.ylabel('Productivity', fontsize = 15) # Defining the Y-Label
plt.legend(loc='upper left') # To show the legend on the plot
plt.savefig("Rainfall Vs Productivity.jpeg",dpi=300) # To save the file in jpeg format with desired resolution
plt.show() # To show the plot

# Plotting the corresponding line over the original data
plt.figure(figsize=(8,6)) # To Create the figure
plt.plot(X, y_predictions, color = 'blue', label='Linear Regression Line') # Plotting the Regression line over the data
plt.scatter(rainfall, productivity, color ='red',edgecolor = "black", label='Rainfall Vs productivity',s=42) # Plotting the scatter plot 
plt.scatter (245, y_pred,c='green',marker='P',edgecolor = "black", label ='Predicted point',s=42) # Plotting the Predicted point
plt.xlabel('Rainfall',fontsize=15) # Defining the x label
plt.ylabel('Productivity',fontsize=15) # Defining the y label
plt.text(60, .165, x, fontsize=10) # Printing the rainfall value on the plot
plt.text(60, .155, y, fontsize=10) # Printing the productivity coefficient value on the plot
plt.title("Rainfall Vs Productivity",size=20) # Defining the title of the plot
plt.legend(loc='upper left') # To show the legend on the plot
plt.savefig("Rainfall Vs Productivity Regression.jpeg",dpi=300) # To save the plot into jpeg file
plt.show() # To show the plot




