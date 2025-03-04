# import dependencies/libraries
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Import dataset and view first 5 rows
df = pd.read_csv('/Users/aleenabasil/Downloads/Stock-Predictor-main/amazon.csv')
df.head()

# Get info
df.shape
df.describe()
df.info()
df.isnull().sum()

# Reverse the rows so the dates are in order
df_reversed = df.iloc[::-1]

# Reset index to maintain the correct sequential order 
df_reversed.reset_index(drop=True, inplace=True)

# View the reversed df to ensure the order is correct
df_reversed.head()

# Convert 'Close/Last' to a string
df_reversed['Close/Last'] = df_reversed['Close/Last'].astype(str)

# Strip dollar signs and convert to numeric
df_reversed['Close/Last'] = df_reversed['Close/Last'].str.replace('$', '', regex=False).astype(float)

# Plot close prices
plt.figure(figsize=(10,5))
plt.plot(df_reversed['Date'], df_reversed['Close/Last'])
plt.title('Amazon Close Price over Time', fontsize = 15)
plt.xlabel('Date')
plt.ylabel('Close/Last Price Price in Dollars')
plt.xticks(rotation=45)
plt.tight_layout()
plt.xticks(ticks=range(0, len(df_reversed['Date']), 100), rotation=45)
plt.show()

# Plot Volume over time
plt.figure(figsize=(10, 5))
plt.plot(df_reversed['Date'], df_reversed['Volume'], label='Volume', color='pink')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Volume Over Time')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(alpha=0.3)
plt.xticks(ticks=range(0, len(df_reversed['Date']), 100), rotation=45)
plt.show()


# Convert 'High' and 'Low' to strings, remove the dollar sign, and then convert to float
df_reversed['High'] = df_reversed['High'].astype(str).str.replace('$', '').astype(float)
df_reversed['Low'] = df_reversed['Low'].astype(str).str.replace('$', '').astype(float)

# Calculate High-Low spread
df_reversed['High/Low Spread'] = df_reversed['High'] - df_reversed['Low']

# Plot High-Low spread over time
plt.figure(figsize=(10, 5))
plt.plot(df_reversed['Date'], df_reversed['High/Low Spread'], label='High-Low Spread', color='orange')
plt.xlabel('Date')
plt.ylabel('High-Low Spread')
plt.title('High-Low Spread Over Time')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.xticks(ticks=range(0, len(df_reversed['Date']), 100), rotation=45)
plt.show()


# Calculate the daily percentage change in 'Close/Last' price
df_reversed['Pct Change'] = df_reversed['Close/Last'].pct_change() * 100  # Percentage change in %

# Plot histogram of daily percentage price change
plt.figure(figsize=(10, 6))
plt.hist(df_reversed['Pct Change'].dropna(), bins=50, color='green', alpha=0.7, edgecolor='black')
plt.title('Distribution of Daily Percentage Price Change')
plt.xlabel('Daily Percentage Price Change (%)')
plt.ylabel('Frequency')
plt.show()


# Reverse the rows so the dates are in order
df = df.iloc[::-1].reset_index(drop=True)


# Convert 'Close/Last', 'High', and 'Low' to float (removing dollar signs)
df['Close/Last'] = df['Close/Last'].str.replace('$', '', regex=False).astype(float)
df['High'] = df['High'].str.replace('$', '', regex=False).astype(float)
df['Low'] = df['Low'].str.replace('$', '', regex=False).astype(float)

# Create a new feature: High-Low spread
df['High/Low Spread'] = df['High'] - df['Low']

# Shift Close/Last to create a 'Future Price' column (target variable)
df['Future Price'] = df['Close/Last'].shift(-1)

# Drop last row with NaN target value
df.dropna(inplace=True)


# Define features and target
features = ['Close/Last', 'High', 'Low', 'High/Low Spread']
target = 'Future Price'


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# Number of future days to predict
future_days = 30  

# Start with the last known data point
future_inputs = df[features].iloc[-1].values.reshape(1, -1)

# Extend x-axis for future days
future_x = np.arange(len(y_test), len(y_test) + future_days)

# Plot actual vs predicted prices with future predictions
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, label='Actual Prices', color='blue')
plt.plot(range(len(y_test)), y_pred, label='Predicted Prices', color='red', linestyle='dashed')
plt.plot(future_x, future_predictions, label='Future Predicted Prices', color='green', linestyle='dotted')
plt.xlabel('Days')
plt.ylabel('Stock Price ($)')
plt.title('Random Forest Stock Price Prediction with Future Forecast')
plt.legend()
plt.show()
