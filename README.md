# FarCast

FarCast is a MatLab code to Forecast price of you desired cryptocurrency using the prices and trading volumes of the highest n cryptocurrencies and highest m stocks of specific industries in trading volume.

Required:
JPL7 econometrics library https://www.spatial-econometrics.com/

What it does:

Data Collection
1. collects information about the highest trading stocks and cyptocurrencies.
2. gets daily prices and volumes of thehighest trading stocks and cryptocurrencies.

Preprocessing:
1. Differentiating the time series.
2. Optional removal of seasonal trends, the number of trends needs to be explicitly specified.
3. Interpolation of missing data points in the time series.
4. applies Hodrick-Prescott (HP) of an adjustable one or multiple frequencies.
5. optional addition of cyclic components to the predictor matrix.
3. optional addition of 1 or more moving avergae vectors to the predictors.

Fitting a model:
1. vector autoregressive models:
It could be done using  with option to add the differentiated predictors. Please don't use the differntiatinf option of this code. instead, set farvmode=1. Addition of cyclic componets and moving averages above 2 are known to produce weird results.

2. Deep learning biLSTM network: (matlab's deep learning toolbox is required)

Forecast validation:
If you set the mode to 'test', FarCast will forecast the most recent data points from older data points and output the absolute error in percentage.

Forecasting:
IF you set the mode to 'Forecast', FarCast will forecast future data points based on the most known ones.
