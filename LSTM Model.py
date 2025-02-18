# import dependencies/libraries
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sb

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error


def preprocessing(df):
    """
    Data preprocessing
    """
    # Reverse the rows so the dates are in order
    df_reversed = df.iloc[::-1]

    # Reset index to maintain the correct sequential order 
    df_reversed.reset_index(drop=True, inplace=True)

    # Clean price columns by removing dollar signs and converting to floats
    df_reversed['Close/Last'] = df_reversed['Close/Last'].astype(str).str.replace('$', '', regex=False).astype(float)
    df_reversed['Open'] = df_reversed['Open'].astype(str).str.replace('$', '', regex=False).astype(float)
    df_reversed['High'] = df_reversed['High'].astype(str).str.replace('$', '').astype(float)
    df_reversed['Low'] = df_reversed['Low'].astype(str).str.replace('$', '').astype(float)

    # Reshape the data into numpy array
    close = df_reversed['Close/Last'].values.reshape(-1, 1)

    # Normalize the data (values are now between 0 and 1)
    close_normalized = close / np.max(close)

    # Split data into 20% training and 20% test sets 
    train_size = int(len(close_normalized) * 0.8)
    train_data = close_normalized[:train_size]
    test_data = close_normalized[train_size:]

    return df_reversed, close_normalized, train_data, test_data


# LSTM model function
def lstm_model(units, activation, learning_rate):
    '''
    Builds an LSTM model
    '''
    model = Sequential()
    model.add(LSTM(units = units, activation = activation, input_shape=(1, 1)))
    
    # Dense layer for output
    model.add(Dense(units=1))

    # Adam optimizer for different learning rates
    optimizer = Adam(learning_rate=learning_rate)

    # Mean squared error (mse) for loss function
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

def train_and_tune(train_data, test_data):
    """
    Trains model and performs hyperparameter tuning
    """

    # Define hyperparameters for tuning
    lstm_units = [50, 100, 200]
    lstm_activations = ['relu', 'tanh']
    learning_rates = [0.001, 0.01, 0.1]
    epochs = 100
    batch_size = 32

    # Perform hyperparameter tuning for LSTM model
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
    
    return best_lstm_model, best_rmse

    # # Predict on the entire dataset using the best LSTM model
    # all_lstm_predictions = best_lstm_model.predict(close_normalized[:-1].reshape(-1, 1, 1)).flatten()

    # # Inverse normalize the LSTM predictions
    # all_lstm_predictions = all_lstm_predictions * np.max(close)


# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(model, actual_prices, predictions):
    """Calculate and return model performance metrics"""
    percentage_error = np.abs((actual_prices[1:] - predictions) / actual_prices[1:] * 100)
    return {
        'mean_error': np.mean(percentage_error),
        'last_5_predictions': list(zip(actual_prices[-5:], predictions[-5:]))
    }

def plot_results(actual_prices, predictions):
    """Create and show the prediction visualization"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices[1:], label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title('Amazon Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
   # Load data
    df = pd.read_csv('/Users/clarawei/school/Stock-Predictor/Stock-Predictor/amazon.csv')
    
    # Preprocess
    df_reversed, close_normalized, train_data, test_data = preprocessing(df)
        
    # Get close values from df_reversed for later use
    close = df_reversed['Close/Last'].values.reshape(-1, 1)

    # Train and tune
    best_model, best_rmse = train_and_tune(train_data, test_data)
    
    # Make predictions
    all_lstm_predictions = best_model.predict(close_normalized[:-1].reshape(-1, 1, 1)).flatten()
    all_lstm_predictions = all_lstm_predictions * np.max(close)
    
    # Evaluate
    metrics = evaluate_model(best_model, close.flatten(), all_lstm_predictions)
    
    # Print results
    print(f"Best RMSE: {best_rmse}")
    print(f"Average prediction error: {metrics['mean_error']:.2f}%")
    print("\nLast 5 predictions vs actual:")
    for actual, pred in metrics['last_5_predictions']:
        print(f"Actual: ${actual:.2f}, Predicted: ${pred:.2f}")
        
    # Visualize
    plot_results(close.flatten(), all_lstm_predictions)