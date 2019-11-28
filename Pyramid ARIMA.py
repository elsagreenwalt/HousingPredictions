import os
import pmdarima as pm
import pandas as pd
from pmdarima import auto_arima
from matplotlib import pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy
import numpy
import sys
from chart_studio.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose

# Set working directory
os.chdir('C:/Users/Elsa/PycharmProjects/HousingPredictions')

# Make list of states to iterate through
state_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

# Create date parser
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')

# Register datetime converter
register_matplotlib_converters()

for state in state_list:
    # Indexing and creating series
    df = pd.read_csv(f'state_csvs/{state}.csv', parse_dates=['Date'], date_parser=dateparse, index_col=0)
    data = df[['Median_Listing_Price']]

    # Divide data into training and testing data
    train = data.loc['2013-11':'2017-01']
    valid = data.loc['2017-02':]

    # Build model
    model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                       trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(train)

    forecast = model.predict(n_periods=len(valid))
    forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

    # predictions = model.forecast(n_periods=10)
    print()

    # Plot the predictions for validation set
    # plt.plot(train, label='Train')
    # plt.plot(valid, label='Valid')
    # plt.plot(forecast, label='Prediction')
    # plt.title(f'{state}')

    # Calculate RMSE
    rms = sqrt(mean_squared_error(valid, forecast))
    print(rms)

    print(model.summary())

    # Future predictions
    model_future = SARIMAX(data, order=(2, 1, 2), seasonal_order=(0,1,0,12))
    results_future = model_future.fit()
    predictions_future = results_future.predict(len(df), len(df) + 240, typ='levels')

    # Plot data
    df.plot(legend=True, figsize=(12, 8), title=f'{state}')
    predictions_future.plot(legend=True, label='Forecasted Median_Listing_Price')
    plt.show()

    # Create dataframe of predictions to add to predictions.csv
    predictions_df = pd.DataFrame(predictions_future)
    predictions_future.columns = ['Date', 'Forecasted Median_Listing_Price']
    predictions_df['Name'] = f'{state}'

    # Write predictions to .csv
    with open('predictions.csv', 'a', newline='') as f:
        predictions_df.to_csv(f)

    print()
# Created auto-arima
