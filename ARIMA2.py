import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt

state_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

def parser(x):
    return datetime.strptime(x, '%Y-%m')

for state in state_list:

    # Create series
    housing_prices = pd.read_csv(f'state_csvs/{state}.csv', header=0, parse_dates=[0], date_parser=parser)

    # Make stationary, integrated order of 1
    housing_prices_diff = housing_prices['Median_Listing_Price'].diff(periods=1)
    housing_prices_diff.plot()
    plt.show()

    # Split into train/test data
    X = housing_prices['Date'].values
    print(X)
    train_size = round(.8 * X.size)
    train = X[0:train_size]
    test = X[train_size:]

    print(X.size)

    print('')
