import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow
from sklearn.preprocessing import MinMaxScaler

Dropout = tensorflow.keras.layers.Dropout
Dense = tensorflow.keras.layers.Dense
LSTM = tensorflow.keras.layers.LSTM

company = 'TSLA'
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)
scaler = MinMaxScaler(feature_range = (0, 1))
scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
predictionDays = 60

xTrain = []
yTrain = []

for x in range(predictionDays, len(scaledData)):
    xTrain.append(scaledData[x - predictionDays: x, 0])
    yTrain.append(scaledData[x, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

model = tensorflow.keras.models.Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (xTrain.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(xTrain, yTrain, epochs = 25, batch_size = 32)

testStart = dt.datetime(2020, 1, 1)
testEnd = dt.datetime.now()
testData = web.DataReader(company, 'yahoo', testStart, testEnd)
actualPrice = testData['Close'].values

totalData = pd.concat((data['Close'], testData['Close']), axis = 0)

modelInputs = totalData[len(totalData) - len(testData) - predictionDays:].values
modelInputs = modelInputs.reshape(-1, 1)
modelInputs = scaler.transform(modelInputs)

xTest = []

for x in range(predictionDays, len(modelInputs)):
    xTest.append(modelInputs[x - predictionDays:x, 0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

predictedPrice = model.predict(xTest)
predictedPrice = scaler.inverse_transform(predictedPrice)

plt.plot(actualPrice, color = 'blue', label = f'actual {company} price')
plt.plot(predictedPrice, color = 'red', label = f'predicted {company} price')
plt.title(f'{company} Share Prices')
plt.xlabel('Time')
plt.ylabel(f'{company} closing Share Price')
plt.legend()
plt.show()

realData = [modelInputs[len(modelInputs) + 1 - predictionDays: len(modelInputs + 1), 0]]
realData = np.array(realData)
realData = np.reshape(realData, (realData.shape[0], realData.shape[1], 1))

prediction = model.predict(realData)
prediction = scaler.inverse_transform(prediction)
print(f'Predicted price is {prediction}')
