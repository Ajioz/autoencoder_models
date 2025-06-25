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
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmdarima.arima import ADFTest
import seaborn as sns
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

plt.plot(df['Passengers'])
plt.title("Monthly Air Passengers")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.show()

# Is the data stationary?
adf_test = ADFTest(alpha=0.05)
print("ADF Test Result:", adf_test.should_diff(df))  # Should differ -> not stationary

# Dickey-Fuller test
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df)
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")

# Extract and plot trend, seasonal and residuals.
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

# Use auto_arima to suggest the best model
arima_model = auto_arima(df['Passengers'], start_p=1, d=1, start_q=1,
                         max_p=5, max_q=5, max_d=5, m=12,
                         start_P=0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5,
                         seasonal=True,
                         trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True, n_fits=50)

print(arima_model.summary())
# Best model from output: SARIMAX(0, 1, 1)x(2, 1, 1, 12)

# Split data into train and test
size = int(len(df) * 0.66)
X_train, X_test = df[0:size], df[size:len(df)]

# Fit SARIMAX model
model = SARIMAX(X_train['Passengers'],
                order=(0, 1, 1),
                seasonal_order=(2, 1, 1, 12))

# Fit model -> The Training Phase
result = model.fit()
print(result.summary())

# In-Sample Prediction (Evaluating the Fit)
train_prediction = result.predict(start=0, end=len(X_train)-1)

# Out-of-Sample Prediction (Evaluating the Forecast)
prediction = result.predict(start=len(X_train), end=len(df)-1).rename('Predicted passengers')

# Plot predictions
prediction.plot(legend=True)
X_test['Passengers'].plot(legend=True)
plt.title("Test Predictions vs Actuals")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.show()

# RMSE
trainScore = math.sqrt(mean_squared_error(X_train['Passengers'], train_prediction))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(X_test['Passengers'], prediction))
print('Test Score: %.2f RMSE' % (testScore))

# Forecast next 3 years
forecast = result.predict(start=len(df), end=(len(df)-1) + 3*12).rename('Forecast')

plt.figure(figsize=(12, 8))
plt.plot(X_train['Passengers'], label='Training', color='green')
plt.plot(X_test['Passengers'], label='Test', color='yellow')
plt.plot(forecast, label='Forecast', color='cyan')
plt.legend(loc='upper left')
plt.title("Forecast for Next 3 Years")
plt.xlabel("Date")
plt.ylabel("Passengers")
plt.show()
