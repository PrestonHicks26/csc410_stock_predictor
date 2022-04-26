import pandas as pd
import numpy as np
from pandas import datetime
from sklearn.model_selection import train_test_split
import pmdarima as pm
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
pd.set_option('display.max_columns', 15)


def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

def forecast_one_step():
    forecast, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        forecast.tolist()[0],
        np.asarray(conf_int).tolist()[0])


data = pd.read_csv('combined.csv', index_col='Date', parse_dates=['Date'], date_parser=parser)
selected_stock = data[data.Company == 'TGT']
company_name = selected_stock.Company[1]
selected_stock = selected_stock.drop('Company', 1)

features = [x for x in selected_stock.columns if x not in ['Increase']]
X = np.array(selected_stock[features])
y = np.array(selected_stock['Increase'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, shuffle=False)

kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss')
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf')
n_diffs = max(adf_diffs, kpss_diffs)
print('Estimated differencing term: ', n_diffs)

auto_vals = pm.auto_arima(y_train, d=n_diffs, easonal=False, stepwise=True,
                      suppress_warnings=True, error_action="ignore", max_p=6, max_order=None, trace=True)

print(auto_vals.order)
#print(model.summary())

model = auto_vals



forecasts = []
confidence_intervals = []

for new_object in y_test:
    fc, conf = forecast_one_step()
    forecasts.append(fc)
    confidence_intervals.append(conf)

    # Updates the existing model with a small number of MLE steps
    model.update(new_object)

print('MSE:', mean_squared_error(y_test, forecasts))
print('SMAPE:', smape(y_test, forecasts))