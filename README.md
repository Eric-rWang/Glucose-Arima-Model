# Glucose-Arima-Model
Glucose Arima model to predict glucose values.


The Arima model is an acronym that stands for...<br/>
AR: Autoregression. A model that uses the current observation and a certain amount of lag observations.<br/>
I: Integrated. Finding the difference between current obervation and an observation at a previous time.<br/>
MA: Moving Average. Uses dependency between an observation and a residual error from a moving average model and applied to lagged observations.

Parameters of Arima Model:<br/>
p: Number of lag observations observed in the model.<br/>
d: Number of times the raw observations are differenced (stationary time series).<br/>
q: Size of moving average window.

The program reads in a cleanned csv file in this case patient1.csv. The series values are split into two groups, training and testing arrays. A rolling forecast Arima model is used to make predictions as the time steps progress. The model adds the predicted data back into the history array and takes into account that data as it makes the next prediction using the model. 
