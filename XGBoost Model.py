#Import dependencies/libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

#Read/get data info
df = pd.read_csv('/Stock-Predictor/amazon.csv')
df.head()
df.shape()
df.describe()
df.info()

#Cleaning data
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

#Determine price columns
price_columns = ["Open", "High", "Low", "Close/Last"]

#Categorize columns
inputs = ["Volume", "Open", "High", "Low"]
target = ["Close/Last"]
x = df[inputs]
y = df[target]
y = y.values.ravel() #Flatten y (numpy array) into 1-D array

#Remove dollar sign and convert to float
df_reversed[price_columns] = df_reversed[price_columns].replace(r'\$', '', regex=True).astype(float)

#Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Scale data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Create the XGBoost model
xgb_model = XGBRegressor(objective="reg:squarederror")

#Define and tune hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300] #number of decision trees in model (larger = better learning/more overfitting risk)
    'learning_rate': [0.01, 0.01, 0.1] #rate at which weights are updated (larger = faster/more overfitting risk)
    'max_depth': [3, 5, 7] #maximum depth of trees in model (larger = more complex/more overfitting risk)
}

#Find optimal hyperparameters using GridSearchCV
grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy')
grid_search.fit(x_train, y_train)

#Make predictions using optimal hyperparameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

#Plot predicted values vs. actual values
plt.figure(figsize=(10,5))
plt.plot( x_test.index, y_test, label="Actual Prices", color='blue', linewidth=2)
plt.plot( x_test.index, y_pred, label="Predicted Prices", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("XGBoost for Stock Price Prediction")
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
accuracy = best_model.score

#Prints optimal set of hyperparameters
print('Optimal set of hyperparameters: ', grid_search.best_params_)

#Prints model evaluation details
print('Mean absolute error: ', mae)
print('Mean squared error: ', mse)
print('Root mean squared error: ', rmse)
print('Model accuracy score: ', accuracy)
