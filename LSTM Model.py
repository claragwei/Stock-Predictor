# import dependencies/libraries
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sb
import tensorflow as tensorflow

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

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam


from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


if __name__ == "__main__":
    # print(df_reversed.head())
    # print(train_data)