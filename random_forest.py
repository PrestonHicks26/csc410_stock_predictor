import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

pd.set_option('display.max_columns', 15)


def sort_by_date(df):
    # method requires a column named Date in the df
    df = df.sort_values(by='Date')
    return df


# -------------------------------------------------------------------------------------------------------------------- #

data = pd.read_csv('combined.csv', parse_dates=['Date'])
data = sort_by_date(data)
print(data.head())
print(data.shape)

selected_stonk = data[data.Company == 'TGT']
selected_stonk = selected_stonk.drop('Company', 1)
print(selected_stonk.head())
print(selected_stonk.shape)

print(selected_stonk.iloc[: , :1])

# TimeSeriesSplit is a variation of the k-fold technique for cross valid. time series data

# custom testing and training split
tscv = TimeBasedCV(train_period=30, test_period=7, freq='days')
features = [x for x in selected_stonk.columns if x not in ['Increase']]
y = selected_stonk['Increase']
X = selected_stonk[features]
print(X.dtypes)
scores = []

for train_index, test_index in tscv.split(selected_stonk, date_column='Date'):
    X_train = X.loc[train_index].drop('Date', axis=1)
    y_train = y.loc[train_index]
    X_test = X.loc[test_index].drop('Date', axis=1)
    y_test = y.loc[test_index]
    # print(train_index, test_index)
    # insert ML algos into here

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    scores.append(score)

avg_model_score = np.mean(scores)
print(avg_model_score)

#rf_model = _train_random_forrest(X_train, y_train, X_test, y_test)

print(tscv.get_n_splits())
