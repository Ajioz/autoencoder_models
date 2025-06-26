"""
Dataset from: https://www.kaggle.com/rakannimer/air-passengers
International Airline Passengers prediction problem.
This is a problem where, given a year and a month, the task is to predict 
the number of international airline passengers in units of 1,000. 
The data ranges from January 1949 to December 1960, or 12 years, with 144 observations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Bidirectional,Input,LSTM, Flatten

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from keras.callbacks import EarlyStopping

# load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
air_passengerPath = os.path.join(BASE_DIR, 'data/AirPassengers.csv')

dataframe = pd.read_csv(air_passengerPath)

if dataframe.empty:
    raise FileNotFoundError(f"DataFrame is empty. Check the file path: {air_passengerPath}")

plt.plot(dataframe['Passengers'])
plt.title("Monthly Air Passengers")
plt.ylabel("Passengers")

# Select only the 'Passengers' column for the model
dataset = dataframe[['Passengers']].values
dataset = dataset.astype('float32') #COnvert values to float

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) #Also try QuantileTransformer
dataset = scaler.fit_transform(dataset)

#We cannot use random way of splitting dataset into train and test as
#the sequence of events is important for time series.
#So let us take first 60% values for train and the remaining 1/3 for testing
# split into train and test sets
train_size = int(len(dataset) * 0.66)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# We cannot fit the model like we normally do for image processing where we have
#X and Y. We need to transform our data into something that looks like X and Y values.
# This way it can be trained on a sequence rather than indvidual datapoints. 
# Let us convert into n number of columns for X where we feed sequence of numbers
#then the final column as Y where we provide the next number in the sequence as output.
# So let us convert an array of values into a dataset matrix

#seq_size is the number of previous time steps to use as 
#input variables to predict the next time period.

#creates a dataset where X is the number of passengers at a given time (t, t-1, t-2...) 
#and Y is the number of passengers at the next time (t + 1).

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)
    

seq_size = 10  # Number of time steps to look back 
#Larger sequences (look further back) may improve forecasting.

trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)



print("Shape of training set: {}".format(trainX.shape))
print("Shape of test set: {}".format(testX.shape))


######################################################
# Reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# print('Single LSTM with hidden Dense...')

# model = Sequential()
# model.add(Input(shape=(None, seq_size)))  # Input shape is (timesteps, features)
# model.add(LSTM(64))
# model.add(Dense(32))
# model.add(Dense(1))

# model.compile(
#     loss=tf.keras.losses.MeanSquaredError(),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     metrics=['mae']
# )

## model.compile(loss='mean_squared_error', optimizer='adam')
##monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, 
#                        verbose=1, mode='auto', restore_best_weights=True)
# model.summary()
# print('Train...')
#########################################

#Stacked LSTM with 1 hidden dense layer
# reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# model = Sequential()
# model.add(Input(shape=(None, seq_size)))  # Input shape is (timesteps, features)
# model.add(LSTM(50, activation='tanh', return_sequences=True))
# model.add(LSTM(50, activation='tanh'))
# model.add(Dense(32, activation='relu'))  # Hidden dense layer
# model.add(Dense(1, activation='linear'))  # Output layer for regression

# model.compile(
#     loss=tf.keras.losses.MeanSquaredError(),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     metrics=['mae']
# )

# model.summary()
# print('Train...')
###############################################

#Bidirectional LSTM
# reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# #For some sequence forecasting problems we may need LSTM to learn
# # sequence in both forward and backward directions
# model = Sequential()
# model.add(Input(shape=(None, seq_size)))  # Input shape is (timesteps, features)
# model.add(Bidirectional(LSTM(50, activation='tanh')))
# model.add(Dense(1))

# model.compile(
#     loss=tf.keras.losses.MeanSquaredError(),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     metrics=['mae']
# )

# model.summary()
# print('Train...')

#########################################################
#ConvLSTM
#The layer expects input as a sequence of two-dimensional images, 
#therefore the shape of input data must be: [samples, timesteps, rows, columns, features]

trainX = trainX.reshape((trainX.shape[0], 1, 1, 1, seq_size))
testX = testX.reshape((testX.shape[0], 1, 1, 1, seq_size))

model = Sequential()
model.add(Input(shape=(1, 1, 1, seq_size)))  # Input shape is (timesteps, features)
model.add(ConvLSTM2D(filters=64, kernel_size=(1,1), activation='tanh'))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(1))

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae']
)

model.summary()
print('Train...')


###############################################
model.fit(trainX, trainY, validation_data=(testX, testY),
          verbose=2, epochs=100)


# make predictions

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions back to prescaled values
#This is to compare with original input values
#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
#we must shift the predictions so that they align on the x-axis with the original dataset. 

# Create empty prediction arrays
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan

# Shift train predictions for plotting
trainPredictPlot[seq_size:seq_size+len(trainPredict), :] = trainPredict

# Shift test predictions for plotting
# First index of test in full dataset = train_size (95)
# After sequencing, predictions start from train_size + seq_size
test_start = train_size + seq_size
testPredictPlot[test_start:test_start + len(testPredict), :] = testPredict


# plot baseline and predictions
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(dataset), label='Original Data')
plt.plot(trainPredictPlot, label='Train Prediction')
plt.plot(testPredictPlot, label='Test Prediction')
plt.title("Air Passenger Forecasting with LSTM")
plt.xlabel("Time (Months)")
plt.ylabel("Passengers")
plt.legend()
plt.show()
