import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


df = pd.read_csv('sp500_stocks.csv')
df = df[df.Date >= '2017-01-01']

index = pd.read_csv('sp500_index.csv')
companies = pd.read_csv('sp500_companies.csv')
# default date starts from Jan 1 of 2017 since 5 years seems reasonable but this can be changed
print(df.head())

user_input = input('Which stock do you want to analyze?\nExample, for Google type GOOG.\nEnter selection here: ')
user_input = user_input.upper()
selected_stock = df[df.Symbol == user_input]

fig = make_subplots(specs=[[{'secondary_y':True}]])
fig.add_trace(
    go.Scatter(x=selected_stock.Date, y=selected_stock.Close, name='Market Close')
)
fig.add_trace(
    go.Scatter(x=selected_stock.Date, y=selected_stock.High, name='Market High')
)
fig.add_trace(
    go.Scatter(x=selected_stock.Date, y=selected_stock.Low, name='Market Low')
)
fig.add_trace(
    go.Scatter(x=selected_stock.Date, y=selected_stock.Open, name='Market Open')
)
fig.update_layout(xaxis_title='Date', yaxis_title='Price', title='User Select Sample Plot')
fig.show()
#
# Analyzing entire SMP 500
#
#
# Machine learning for data
#


array = selected_stock.values
X = array[:,0:4]
y = array[:,4]                  # values need to be changed to the finalized dataset
#
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# add more models here in the future
names = []
results = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

   
# program not completed yet and does not show any output besides initial stock visualization; need to improve datasets and tweak ML application
