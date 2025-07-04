"""
Anomaly Detection using LSTM Autoencoder

Author: Sreenivas Bhattiprolu
Fixed for compatibility and pandas error by OpenAI ChatGPT

Dataset: GE stock data from Yahoo Finance (https://finance.yahoo.com/quote/GE/history/)
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
gePath = os.path.join(BASE_DIR, 'data/GE.csv')

# Read CSV and normalize column names
dataframe = pd.read_csv(gePath)
dataframe.columns = dataframe.columns.str.strip().str.lower()  # normalize to lowercase

# Validate required columns
required_cols = ['date', 'close']
if not all(col in dataframe.columns for col in required_cols):
    raise ValueError(f"Missing required columns. Found: {dataframe.columns.tolist()}")

# Extract relevant columns and rename for consistency
df = dataframe[['date', 'close']].copy()
df.rename(columns={'date': 'Date', 'close': 'Close'}, inplace=True)
# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Visualize raw stock data
sns.lineplot(x=df['Date'], y=df['Close'])
plt.title('GE Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

print("Start date is:", df['Date'].min())
print("End date is:", df['Date'].max())

# Split into train/test sets
train = df[df['Date'] <= '2003-12-31'].copy()
test = df[df['Date'] > '2003-12-31'].copy()

# Normalize the 'Close' price
scaler = StandardScaler()
train['Close'] = scaler.fit_transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])

# Convert to sequences for LSTM
def to_sequences(x, y, seq_size=1):
    x_values, y_values = [], []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)

seq_size = 30  # Number of timesteps
trainX, trainY = to_sequences(train[['Close']], train['Close'], seq_size)
testX, testY = to_sequences(test[['Close']], test['Close'], seq_size)

# LSTM Autoencoder model
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(RepeatVector(trainX.shape[1]))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

# Train model
history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Plot training loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Get reconstruction loss (MAE)
trainPredict = model.predict(trainX)
trainMAE = np.mean(np.abs(trainPredict - trainX), axis=(1, 2))
plt.hist(trainMAE, bins=30)
plt.title('Training MAE Distribution')
plt.show()

max_trainMAE = 0.3  # Can be tuned or use percentile-based threshold

testPredict = model.predict(testX)
testMAE = np.mean(np.abs(testPredict - testX), axis=(1, 2))
plt.hist(testMAE, bins=30)
plt.title('Test MAE Distribution')
plt.show()

# Mark anomalies
anomaly_df = test.iloc[seq_size:].copy()
anomaly_df['testMAE'] = testMAE
anomaly_df['max_trainMAE'] = max_trainMAE
anomaly_df['anomaly'] = anomaly_df['testMAE'] > max_trainMAE

# Plot anomaly scores
sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['testMAE'], label='Test MAE')
sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['max_trainMAE'], label='Threshold')
plt.title('Anomaly Detection Threshold vs Test MAE')
plt.xlabel('Date')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Plot anomalies
anomalies = anomaly_df[anomaly_df['anomaly'] == True]

close_prices = scaler.inverse_transform(anomaly_df[['Close']])
anomaly_df['Close_Inverse'] = close_prices
anomalies['Close_Inverse'] = scaler.inverse_transform(anomalies[['Close']])

sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['Close_Inverse'], label='Close Price')
sns.scatterplot(x=anomalies['Date'], y=anomalies['Close_Inverse'], color='r', label='Anomaly')
plt.title('Anomalies in GE Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
