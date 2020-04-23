import csv
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


# reading in the csv file
glu = pd.read_csv('patient1.csv', header = 0, index_col = [0], parse_dates = [0], squeeze = True)

# prints out the head of csv file
print(glu.head(), str('\n'))

# limiting glu array to only patient ID 1
glu = glu[0:1482]
print(glu.describe())

# array containing glu values at each time
series_value = glu.values
print(type(series_value), str('\n')) # confirming series value type as darray

# smoothing out curve
glu_mean = glu.rolling(window = 10).mean()

# plotting the glu_mean (rolling average plot)
glu_mean.plot()
plt.show()


# creating baseline algorithm, niave solution (test)
value = pd.DataFrame(series_value)
glu_df = pd.concat([value, value.shift(1)], axis = 1)

glu_df.columns = ['Actual_glu', 'Forecast_glu']
print(glu_df.head(), str('\n'))

# error calculation
glu_test = glu_df[1:]
print(glu_test.head(), str('\n'))

glu_error = mean_squared_error(glu_test.Actual_glu, glu_test.Forecast_glu)
print(np.sqrt(glu_error), str('\n'))

# patient ID produced error of 1.527598902356933, other models should be below...


# ARIMA - Autoregressive Integrated Moving Average model
plot_acf(glu) # identify value of q
plot_pacf(glu) # identify value of p
plt.show()

print()
print(glu.size, str('\n'))

# train and test
size = int(len(glu) * 0.66)
glu_train = glu[0:size]
glu_test = glu[size:len(glu)]

history = [glu for glu in glu_train]
history.remove(history[0])

for i in range(len(history)):
	history[i] = int(history[i])

autocorrelation_plot(glu)
plt.show()

# fit model
'''
glu_model = ARIMA(history, order = (5, 0, 1))
glu_model_fit = glu_model.fit(disp = 0)
print(glu_model_fit.summary())

residuals = pd.DataFrame(glu_model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind = 'kde')
plt.show()
print(residuals.describe())
'''

predictions = list()
# only prediction 20 increments into the future
test = glu_test[0:100]

# rolling forcast model
for t in range(len(test)):
	glu_model = ARIMA(history, order = (5, 0, 1))
	glu_model_fit = glu_model.fit(disp = 0)
	output = glu_model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = glu_test[t]
	history.append(yhat)
	print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.f' % error)

# plot
plt.plot(test)
plt.show()

plt.plot(predictions, color = 'red')
plt.show()

# writing out to csv file
with open('results.csv', 'w') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(["expected", "predicted"])

	for i in range(len(predictions)):
		csvwriter.writerow([test[i], predictions[i][0]])







