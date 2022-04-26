import pandas as pd
import datetime
from pandas import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *
import numpy as np
from time_series_cross_validation import TimeBasedCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pmdarima as pm
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
pd.set_option('display.max_columns', 15)


def parser(x):
	return datetime.strptime(x, '%Y-%m-%d')

def sort_by_date(df):
    # method requires a column named Date in the df
    df = df.sort_values(by='Date')
    return df

def forecast_one_step():
    forecast, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        forecast.tolist()[0],
        np.asarray(conf_int).tolist()[0])

def train_random_forest(X_train, y_train, X_test, y_test):

    '''
    Function that uses random forrest classifier to train the model
    :return:
    '''

    # creating classifier
    rf = RandomForestClassifier()

    # dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [10, 50,110,130,140,150,160,180, 200]}

    # use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)

    # fit model to training data
    rf_gs.fit(X_train, y_train)

    # save the best model
    rf_best = rf_gs.best_estimator_

    # check best n_estimator value
    print(rf_gs.best_params_)

    prediction = rf_best.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))

    return rf_best


data_arima = pd.read_csv('combined.csv', index_col='Date', parse_dates=['Date'], date_parser=parser)
selected_stock = data_arima[data_arima.Company == 'TGT']
company_name = selected_stock.Company[1]
selected_stock = selected_stock.drop('Company', 1)
datasets = []
datasets.append(data_arima)

rf_sklearn_score = []
models = []

features = [x for x in selected_stock.columns if x not in ['Increase']]
X = np.array(selected_stock[features])
y = np.array(selected_stock['Increase'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, shuffle=False)
rf_model = train_random_forest(X_train, y_train, X_test, y_test)
rf_prediction = rf_model.predict(X_test)
print('Random forest with train-test-split prediction: ', rf_prediction)
score = accuracy_score(y_test, rf_prediction)
print(score)
rf_sklearn_score.append(score)
print('RF Accuracy using train-test-split: ' + str( sum(rf_sklearn_score) / len(rf_sklearn_score)))
# RF Accuracy = 0.4583333333333333
avg_model_score = np.mean(rf_sklearn_score)
print('Random Forest with SKlearn train-test-split Average score: ',avg_model_score)