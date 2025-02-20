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
from sklearn.preprocessing import MinMaxScaler


def preprocessing(df):
    """
    Data preprocessing to clean and normalize stock price
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
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_normalized = scaler.fit_transform(close)

    # Split data into 80% training and 20% test sets 
    train_size = int(len(close_normalized) * 0.8)
    train_data = close_normalized[:train_size]
    test_data = close_normalized[train_size:]

    return df_reversed, close_normalized, train_data, test_data, scaler


# LSTM model function
def lstm_model(units, activation, learning_rate):
    '''
    Builds an LSTM model
    '''
    model = Sequential()
    model.add(LSTM(units = units, activation = activation, input_shape=(1, 1)))
    
    # Dropout layer to prevent overfitting
    model.add(Dropout(0.2))

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

def rmse(y_true, y_pred):
    """
    Function to calculate RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def predict_future(model, last_known, days, scaler):
    """
    Generates future predictions based on the trained model
    """
    predictions = []
    current_input = last_known.reshape(1, 1, 1)

    for _ in range(days):
        predicted_value = model.predict(current_input)[0, 0]
        predictions.append(predicted_value)
        current_input = np.array([[predicted_value]]).reshape(1, 1, 1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def evaluate_model(model, actual_prices, predictions):
    """
    Calculate and return model performance metrics
    """
    percentage_error = np.abs((actual_prices[1:] - predictions) / actual_prices[1:] * 100)
    return {
        'mean_error': np.mean(percentage_error),
        'last_5_predictions': list(zip(actual_prices[-5:], predictions[-5:]))
    }

def plot_results(actual_prices, predictions, future_predictions = None):
    """
    Create and show actual vs predicted prices
    """
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices[1:], label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.axvline(x=len(predictions), color='r', linestyle='--', label='Future Predictions Start')
    plt.plot(range(len(predictions), len(predictions) + len(future_predictions)), future_predictions, label='Future Predictions', linestyle='dashed')
    plt.title('Amazon Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
   # Load data
    df = pd.read_csv('/Users/clarawei/school/Stock-Predictor/Stock-Predictor/amazon.csv')
    
    # Preprocess
    df_reversed, close_normalized, train_data, test_data, scaler = preprocessing(df)
        
    # Get close values from df_reversed for later use
    close = df_reversed['Close/Last'].values.reshape(-1, 1)

    # Train and tune
    best_model, best_rmse = train_and_tune(train_data, test_data)
    
    # Make predictions
    all_lstm_predictions = best_model.predict(close_normalized[:-1].reshape(-1, 1, 1)).flatten()
    all_lstm_predictions = scaler.inverse_transform(all_lstm_predictions.reshape(-1, 1)).flatten()
    
    # Predict on next 30 days
    future_steps = 30
    future_predictions = predict_future(best_model, close_normalized, future_steps)
    
    # Evaluate
    metrics = evaluate_model(best_model, close.flatten(), all_lstm_predictions)
    
    future_days = 10
    future_predictions = predict_future(best_model, close_normalized[-1], future_days, scaler)

    # Print results
    print(f"Best RMSE: {best_rmse}")
    print(f"Average prediction error: {metrics['mean_error']:.2f}%")
    print("\nLast 5 predictions vs actual:")
    for actual, pred in metrics['last_5_predictions']:
        print(f"Actual: ${actual:.2f}, Predicted: ${pred:.2f}")

    print(f"Best RMSE: {best_rmse}")
    plot_results(close.flatten(), all_lstm_predictions, future_predictions)
