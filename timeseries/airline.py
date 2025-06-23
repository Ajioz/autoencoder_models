"""
A quick overview of timeseries. 

Dataset from: https://www.kaggle.com/rakannimer/air-passengers
International Airline Passengers prediction problem.
This is a problem where, given a year and a month, the task is to predict 
the number of international airline passengers in units of 1,000. 
The data ranges from January 1949 to December 1960, or 12 years, with 144 observations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

plt.style.use('dark_background')

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
air_passengerPath = os.path.join(BASE_DIR, 'data/AirPassengers.csv')

df = pd.read_csv(air_passengerPath)

if df.empty:
    raise FileNotFoundError(f"DataFrame is empty. Check the file path: {air_passengerPath}")

# Convert date column
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(df['Passengers'], color='yellow')
plt.title('Monthly International Air Passengers')
plt.xlabel('Date')
plt.ylabel('Passengers (in thousands)')
plt.tight_layout()
plt.show()

# Stationarity check: Augmented Dickey-Fuller Test
adf, pvalue, _, _, _, _ = adfuller(df['Passengers'])
print("ADF Test p-value =", pvalue, "-> If above 0.05, data is not stationary")

# Year and month columns for boxplot analysis
df['year'] = df.index.year.map(str)
df['month'] = df.index.strftime('%b')
years = df['year'].unique()

# Boxplot by year
plt.figure(figsize=(10, 5))
sns.boxplot(x='year', y='Passengers', data=df)
plt.title('Yearly Distribution of Passengers')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.tight_layout()
plt.show()

# Boxplot by month
plt.figure(figsize=(10, 5))
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sns.boxplot(x='month', y='Passengers', data=df, order=month_order)
plt.title('Monthly Distribution of Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.tight_layout()
plt.show()

# Seasonal decomposition
decomposed = seasonal_decompose(df['Passengers'], model='additive')
trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df['Passengers'], label='Original', color='yellow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='yellow')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal', color='yellow')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residual', color='yellow')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Autocorrelation with ACF values
acf_vals = acf(df['Passengers'], nlags=40)
plt.figure(figsize=(10, 4))
plt.plot(acf_vals, marker='o', linestyle='-', color='yellow')
plt.title('Autocorrelation (manual)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.tight_layout()
plt.show()

# Preferred autocorrelation plot with confidence intervals
plot_acf(df['Passengers'], lags=40)
plt.title('Autocorrelation with Confidence Intervals')
plt.tight_layout()
plt.show()

# Optional: autocorrelation_plot (less preferred)
# autocorrelation_plot(df['Passengers'])
# plt.show()
