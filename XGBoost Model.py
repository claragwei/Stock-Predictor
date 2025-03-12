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

#Clean data

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

#Determine the inputs and target output
price_columns = ["Open", "High", "Low", "Close/Last"]

#Remove dollar sign and convert to float
df_reversed[price_columns] = df_reversed[price_columns].replace(r'\$', '', regex=True).astype(float)
