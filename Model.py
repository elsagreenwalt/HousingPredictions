import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime as dt

state_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

for state in state_list:
    with open(f'state_csvs/{state}.csv', 'r') as f:

        # Create data frame
        data_df = pd.read_csv(f'state_csvs/{state}.csv')

        # Convert dates to numbers for model
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df['Date'] = data_df['Date'].map(dt.datetime.toordinal)

        # Create x and y arrays
        x = np.array(data_df['Date']).reshape(-1,1)
        y = np.array(data_df['Median_Listing_Price']).reshape(-1,1)

        model = LinearRegression()  # Create linear regression model
        model.fit(x, y)  # Train model
        model.score(x, y)  # Check score

        print('\nCoefficient: \n', model.coef_)
        print('Intercept: \n', model.intercept_)
