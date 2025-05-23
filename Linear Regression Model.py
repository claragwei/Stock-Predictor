#Import dependencies/libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

#Read/get data info
df = pd.read_csv('/Stock-Predictor/amazon.csv')
df.head()
df.shape()
df.describe()
df.info()

#Cleaning the data
#Reverse rows so dates are in order
df_reversed = df.iloc[::-1]

#Reset index to maintain correct sequential order
df_reversed.reset_index(drop=True, inplace=True)

#View reversed df to ensure the order is correct
df_reversed.head()

#Convert the 'Date' column from an object type to a datetime type in a pandas DataFrame
df_reversed['Date'] = pd.to_datetime(df_reversed['Date'])
df_reversed = df_reversed.set_index('Date')
df_reversed = df_reversed.sort_values(by='Date')

#Reshape data
close = df_reversed['Close/Last'].values.reshape(-1, 1)

#Normalize data so values are between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_close = scaler.fit_transform(close)

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Create the Linear Regression model
lg_model = LinearRegression().fit(x_train, y_train)

#Make predictions
y_pred = lg_model.predict(x_test)

#Plot predicted values vs. actual values
plt.figure(figsize=(10,5))
plt.plot( x_test.index, y_test, label="Actual Prices", color='blue', linewidth=2)
plt.plot( x_test.index, y_pred, label="Predicted Prices", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Linear Regression Model for Stock Price Prediction")
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
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
accuracy = lg_model.score

#Prints model evaluation details
print('Mean absolute error: ', mae)
print('Mean squared error: ', mse)
print('Root mean squared error: ', rmse)
print('Model accuracy score: ', accuracy)
