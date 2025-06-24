"""
Airline Passenger Forecasting - Modular Version
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import math

plt.style.use('dark_background')


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    if df.empty:
        raise FileNotFoundError(f"DataFrame is empty. Check the file path: {filepath}")
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df


def test_stationarity(df):
    result = adfuller(df['Passengers'])
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    return result[1] <= 0.05


def decompose_and_plot(df):
    decomposed = seasonal_decompose(df['Passengers'], model='additive')
    trend, seasonal, residual = decomposed.trend, decomposed.seasonal, decomposed.resid

    plt.figure(figsize=(12, 8))
    for i, (data, label) in enumerate(zip([df, trend, seasonal, residual],
                                          ['Original', 'Trend', 'Seasonal', 'Residual'])):
        plt.subplot(4, 1, i + 1)
        plt.plot(data, label=label, color='yellow')
        plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def fit_auto_arima(df):
    model = auto_arima(df['Passengers'], start_p=1, d=1, start_q=1,
                       max_p=5, max_q=5, max_d=5, m=12,
                       start_P=0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5,
                       seasonal=True, trace=True, error_action='ignore',
                       suppress_warnings=True, stepwise=True, n_fits=50)
    print(model.summary())
    return model


def train_test_split(df, ratio=0.66):
    size = int(len(df) * ratio)
    return df[0:size], df[size:]


def fit_sarimax_model(train_data):
    model = SARIMAX(train_data, order=(0, 1, 1), seasonal_order=(2, 1, 1, 12))
    result = model.fit()
    print(result.summary())
    return result


def plot_predictions(train, test, prediction):
    prediction.plot(label='Predicted', color='cyan')
    test['Passengers'].plot(label='Actual', color='yellow')
    plt.title("Test Predictions vs Actuals")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Passengers")
    plt.show()


def calculate_rmse(actual, predicted, label=''): 
    score = math.sqrt(mean_squared_error(actual, predicted))
    print(f'{label} RMSE: {score:.2f}')
    return score


def plot_forecast(train, test, forecast):
    plt.figure(figsize=(12, 8))
    plt.plot(train, label='Training', color='green')
    plt.plot(test, label='Test', color='yellow')
    plt.plot(forecast, label='Forecast', color='cyan')
    plt.legend(loc='upper left')
    plt.title("Forecast for Next 3 Years")
    plt.xlabel("Date")
    plt.ylabel("Passengers")
    plt.show()


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, 'data/AirPassengers.csv')

    df = load_dataset(path)
    print("Stationary:", test_stationarity(df))

    decompose_and_plot(df)

    _ = fit_auto_arima(df)  # Optional: just to show summary

    train, test = train_test_split(df)

    model_result = fit_sarimax_model(train['Passengers'])

    train_pred = model_result.predict(start=0, end=len(train) - 1)
    test_pred = model_result.predict(start=len(train), end=len(df) - 1).rename('Predicted passengers')

    plot_predictions(train, test, test_pred)

    calculate_rmse(train['Passengers'], train_pred, label='Train')
    calculate_rmse(test['Passengers'], test_pred, label='Test')

    forecast = model_result.predict(start=len(df), end=(len(df) - 1) + 3 * 12).rename('Forecast')
    plot_forecast(train, test, forecast)
