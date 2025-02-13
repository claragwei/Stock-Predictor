# import dependencies/libraries
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sb


# Import dataset and view first 5 rows
df = pd.read_csv('/Users/clarawei/school/Stock-Predictor/Stock-Predictor/amazon.csv')
df.head()

# Get info
df.shape()
df.describe()
df.info()
df.isnull().sum()