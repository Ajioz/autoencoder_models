"""
Anomaly Detection using LSTM Autoencoder
Author: Sreenivas Bhattiprolu
modified by: Ajroghene Sunny
Date: 2023-10-30
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

# --- Hyperparameters ---
SEQ_SIZE = 30  # Number of timesteps in a sequence
THRESHOLD_PERCENTILE = 0.99  # The percentile for setting the anomaly threshold. Tune this value based on problem sensitivity.

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
    # Ensure x is a DataFrame for consistent slicing
    if isinstance(x, pd.Series):
        x = x.to_frame()
    x_values, y_values = [], []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)

seq_size = SEQ_SIZE
trainX, trainY = to_sequences(train[['Close']], train['Close'], seq_size)
testX, testY = to_sequences(test[['Close']], test['Close'], seq_size)

# LSTM Autoencoder model
model = Sequential()
# --- Encoder ---
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
# --- Bridge ---
model.add(RepeatVector(trainX.shape[1]))
# --- Decoder ---
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))

model.compile(optimizer='adam', loss='mae')

model.summary()

# Train model
# A standard autoencoder is trained to reconstruct its own input.
history = model.fit(trainX, trainX, epochs=30, batch_size=32, validation_split=0.1, verbose=1)


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

# Set a dynamic threshold based on a percentile of the training MAE.
# This is more robust than a hardcoded value.
max_trainMAE = np.quantile(trainMAE, THRESHOLD_PERCENTILE)
print(f"Anomaly threshold ({(THRESHOLD_PERCENTILE * 100):.0f}th percentile) set to: {max_trainMAE:.4f}")

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
# Create an explicit copy to avoid the SettingWithCopyWarning
anomalies = anomaly_df[anomaly_df['anomaly'] == True].copy()

close_prices = scaler.inverse_transform(anomaly_df[['Close']])
anomaly_df['Close_Inverse'] = close_prices
anomalies['Close_Inverse'] = scaler.inverse_transform(anomalies[['Close']]) # This now safely works on a copy

sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['Close_Inverse'], label='Close Price')
sns.scatterplot(x=anomalies['Date'], y=anomalies['Close_Inverse'], color='r', label='Anomaly')
plt.title('Anomalies in GE Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
