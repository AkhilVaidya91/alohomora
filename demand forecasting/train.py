import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import statsmodels.api as sm
import pickle
import os

## Constants

P = 1
Q = 2
D = 0
S = 52

def load_data(data_path, sales_col_title):
    df = pd.read_csv(data_path)
    data = pd.DataFrame()
    data['week'] = pd.to_datetime(df['week_start_date'])
    data['sales'] = df[sales_col_title]
    return data

def fit_sarimax(data):
    model = sm.tsa.statespace.SARIMAX(data['sales'], order=(P, D, Q), seasonal_order=(P, D, Q, S))
    model_fit = model.fit()
    return model_fit

def fit_arima(data):
    model = sm.tsa.ARIMA(data['sales'], order=(P, D, Q))
    model_fit = model.fit()
    return model_fit

def train_model(data_path, sales_col_title):
    csv_name = data_path.split('/')[-1].split('.')[0]
    data = load_data(data_path, sales_col_title)
    model_fit = fit_sarimax(data)
    # model_fit = fit_arima(data)
    pickle.dump(model_fit, open(f'models/{csv_name}_arima.pkl', 'wb'))

def predict_next_four_weeks(model: SARIMAXResults, start_date: str): ## start_date: 'YYYY-MM-DD'

    start_date = pd.to_datetime(start_date)
    start_week = start_date.isocalendar()[1]
    end_week = start_week + 4

    predictions = model.predict(start=start_week, end=end_week)
    return list(predictions.values)


if __name__ == '__main__':
    
    for data_path in os.listdir('data'):
        filename = data_path.split('.')[0]
        PATH = f'data/{data_path}'
        path = r'{}'.format(PATH)
        train_model(path, filename)