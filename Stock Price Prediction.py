import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

dataset_path = os.getcwd() + "\\Dataset\\AAPL_train.csv"
train = pd.read_csv(dataset_path)
AAPL_stock = train.iloc[:,1:2].values
print(len(AAPL_stock))

scaler = MinMaxScaler(feature_range = (0,1))
scaled_AAPL_stock = scaler.fit_transform(AAPL_stock)

X_train = []
y_train = []

for i in range(90,2265):
    X_train.append(scaled_AAPL_stock[i-90:i, 0])
    y_train.append(scaled_AAPL_stock[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

dataset_test_path = os.getcwd() + "\\Dataset\\AAPL_test.csv"
test = pd.read_csv(dataset_test_path)
actual_AAPL_stock_price = test.iloc[:,1:2].values

total = pd.concat((train['Open'], test['Open']), axis = 0)
inputs = total[len(total)- len(test)-90:].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(90,110):
    X_test.append(inputs[i-90:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

predicted_AAPL_stock_price = model.predict(X_test)
predicted_AAPL_stock_price = scaler.inverse_transform(predicted_AAPL_stock_price)

plt.plot(actual_AAPL_stock_price, color = 'red', label = 'Actual Apple Stock Price')
plt.plot(predicted_AAPL_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()