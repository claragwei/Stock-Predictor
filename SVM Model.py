#Import dependencies/libraries
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('/Stock-Predictor/amazon.csv')
df.head()

#Get data info
df.shape
df.describe()
df.info()

#Cleaning the data
#Reverse the rows so the dates are in order
df_reversed = df.iloc[::-1]

#Reset index to maintain the correct sequential order 
df_reversed.reset_index(drop = True, inplace = True)

#View the reversed df to ensure the order is correct
df_reversed.head()

#Convert the 'Date' column from an object type to a datetime type in a pandas DataFrame
df_reversed['Date'] = pd.to_datetime(df_reversed['Date'])
df_reversed = df_reversed.set_index('Date')
df_reversed = df_reversed.sort_values(by = 'Date')

#Determine the inputs and target output
price_columns = ["Open", "High", "Low", "Close/Last"]

#Remove the dollar sign and convert to float for all price columns
df_reversed[price_columns] = df_reversed[price_columns].replace(r'\$', '', regex=  True).astype(float)

#Categorize the columns based on what we want
inputs = ["Volume", "Open", "High", "Low"]
target = ["Close/Last"]
x = df[inputs]
y = df[target]

#Error from training model... y needs to be flattened with ravel()
y = y.values.ravel()

#Make the training and testing sets for machine learning
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = False)

#Scale the data (this one is small but if for future needs)
scale = MinMaxScaler()
x_train_scaled = scale.fit_transform(x_train)
x_test_scaled = scale.transform(x_test)

#Train the model
svm_model = SVR(kernel = 'rbf', C = 100, gamma = 0.2, epsilon = 0.1) 
#support vector regression
#epsilon-insensitive loss function (ignores errors smaller than a specified threshold)
#kernel rbf (radial basis function) used to capture non-linearity in the data
#C controls how much model tries to fit the data exactly (higher = focus on minimizing errors, 
#lower = more flexibility but doesnt fit as closely to the data)
#gamma controls how much influence a single data point has (high = focus on nearby points, low = consider broader patterns)
svm_model.fit(x_train_scaled, y_train) #model learns from the training data and finds 

#Make predictions
y_pred = svm_model.predict(x_test_scaled)

#Plot the predicted values vs the actual values to see accuracy
plt.figure(figsize=(10,5))
plt.plot( x_test.index, y_test, label = "Actual Prices", color = 'blue', linewidth = 2)
plt.plot( x_test.index, y_pred, label= "Predicted Prices", color = 'red', linestyle = "dashed")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("SVM for Amazon Stock Price Prediction")

#Format the dates
plt.xticks(rotation = 45, ha = 'right', fontsize = 10)
#plt.xticks(ticks = range(0, len(df_reversed['Date']), 100), rotation = 45)
plt.yticks(fontsize = 10)
plt.tight_layout() #Prevents labels from overlapping
plt.legend()
plt.show()

#Evaluate the model using error metrics and accuracy
#Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
#Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
#Root Mean Squared Error
rmse = np.sqrt(mse)

#Test the model for accuracy
#Determines the r^2 score, or coefficient of determination. The closer to 1.0, the more accurate the model
accuracy = svm_model.score(x_test, y_test)

#Prints model evaluation details
print('Mean absolute error: ', mae)
print('Mean squared error: ', mse)
print('Root mean squared error: ', rmse)
print('Model accuracy score: ', accuracy)
