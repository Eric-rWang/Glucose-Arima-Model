import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import numpy as numpy


def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

# load in dataset
data = pd.read_csv('testdata.csv')

print('\n Data Types:')
print(data.dtypes)

X = difference(data.values)
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:]

model = AR(train)
model_fit = model.fit(maxlag = 6, disp = False)
window = model_fit.k_ar
coef = model_fit.params

history = [train[i] for i in range(len(train))]
predictions = list()

for t in range(len(test)):
	yhat = predict(coef, history)
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)

error = mean_squared_error(test, predictions)
print('Test MSE: %0.3f' % error)

pyplot.plot(test)
pyplot.plot(predictions, color = 'red')
pyplot.show()