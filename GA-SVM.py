import seaborn as sns
from genetic_selection import GeneticSelectionCV
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import svm
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import time
from sklearn import metrics


combined_df = pd.read_csv('combined.csv', index_col='Date', parse_dates=['Date'])
ga_df = combined_df.iloc[:2000,:]
x = ga_df.loc[:, ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'H-L', 'O-AdjC', '7 Day MA',
                        '14 Day MA', '21 Day MA', '7 Days Standard Deviation']]
y = ga_df.loc[:, ['Increase']]
estimator = DecisionTreeClassifier()
model = GeneticSelectionCV(estimator)
start = time.time()
model = model.fit(x, np.ravel(y))
end = time.time()
print('GA runtime: ' + str(end-start))
print('Features:', x.columns[model.support_])

svm_df = combined_df.iloc[:500,:]
X = svm_df.loc[:, x.columns[model.support_]]
y = svm_df.loc[:, ['Increase']]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1, shuffle=False)

classifier = svm.SVC(kernel='linear')
start = time.time()
model = classifier.fit(x_train, np.ravel(y_train))
stop = time.time()
print("SVM runtime: " + str(stop - start))
print("training complete")
y_predict = classifier.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
cf_matrix = confusion_matrix(y_test, y_predict)
auc = metrics.roc_auc_score(y_test, y_predict)
print('The AUC score is: ' + str(auc))

fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)

eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
print('The EER is: ' + str(eer))
thresh = interp1d(fpr, thresholds)(eer)

axis = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
axis.set_title('SVM Confusion Matrix')
axis.set_xlabel('Predicted Values')
axis.set_ylabel('Actual Values')
axis.xaxis.set_ticklabels(['False', 'True'])
axis.yaxis.set_ticklabels(['False', 'True'])
plt.show()
