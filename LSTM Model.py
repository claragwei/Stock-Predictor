# import dependencies/libraries
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sb
import tensorflow as tf

# Import dataset and view first 5 rows
df = pd.read_csv('/Users/clarawei/school/Stock-Predictor/Stock-Predictor/amazon.csv')
df.head()

# Get info
# df.shape()
df.describe()
df.info()
df.isnull().sum()

# Reverse the rows so the dates are in order
df_reversed = df.iloc[::-1]

# Reset index to maintain the correct sequential order 
df_reversed.reset_index(drop=True, inplace=True)

# Convert 'Close/Last' to a string, strip dollar signs and convert to float
df_reversed['Close/Last'] = df_reversed['Close/Last'].astype(str).str.replace('$', '', regex=False).astype(float)

# Convert 'Open' to a string, strip dollar signs and convert to float
df_reversed['Open'] = df_reversed['Open'].astype(str).str.replace('$', '', regex=False).astype(float)

# Convert 'High' and 'Low' to strings, remove the dollar sign, and convert to float
df_reversed['High'] = df_reversed['High'].astype(str).str.replace('$', '').astype(float)
df_reversed['Low'] = df_reversed['Low'].astype(str).str.replace('$', '').astype(float)


# Preprocessing

# Reshape the data into numpy array
close = df_reversed['Close/Last'].values.reshape(-1, 1)

# Normalize the data (values are now between 0 and 1)
close_normalized = close / np.max(close)

# Split data into training and test sets 
train_size = int(len(close_normalized) * 0.8) # 80% training, 20% testing
train_data = close_normalized[:train_size]
test_data = close_normalized[train_size:]

# importing necessary libraries
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


from tensorflow.keras.models import Sequential
from tensorflow.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error



# LSTM model function
def lstm_model(units, activation, learning_rate):
    '''
    Builds an LSTM model
    '''
    model = Sequential()
    model.add(LSTM(units = units, activation = activation, input_shape=(1, 1)))
    # dense layer for output
    model.add(Dense(units=1))
    # adam optimizer for different learning rates
    optimizer = Adam(learning_rate=learning_rate)
    # mean squared error (mse) for loss function
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# define hyperparameters for tuning
lstm_units = [50, 100, 200]
lstm_activations = ['relu', 'tanh']
learning_rates = [0.001, 0.01, 0.1]
epochs = 100
batch_size = 32

# perform hyperparameter tuning for LSTM model
best_rmse = float('inf')
best_lstm_model = None

for units in lstm_units:
    for activation in lstm_activations:
        for learning_rate in learning_rates:
            # Create and train LSTM model
            model = lstm_model(units=units, activation=activation, learning_rate=learning_rate)
            model.fit(train_data[:-1].reshape(-1, 1, 1), train_data[1:], epochs=epochs, batch_size=batch_size, verbose=0)

            # Predict on test data
            test_predictions = model.predict(test_data[:-1].reshape(-1, 1, 1)).flatten()

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(test_data[1:], test_predictions))

            # Check if current model has lower RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_lstm_model = model

# Predict on the entire dataset using the best LSTM model
all_lstm_predictions = best_lstm_model.predict(data_normalized[:-1].reshape(-1, 1, 1)).flatten()

# Inverse normalize the LSTM predictions
all_lstm_predictions = all_lstm_predictions * np.max(data)

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


if __name__ == "__main__":
    pass
    # print(df_reversed.head())
    # print(train_data)