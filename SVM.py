import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
import seaborn as sns

combined_df = pd.read_csv('combined.csv', index_col='Date', parse_dates=['Date'])
combined_df = combined_df.iloc[:1000,:]
x = combined_df.loc[:, ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'H-L', 'O-AdjC', '7 Day MA',
                        '14 Day MA', '21 Day MA', '7 Days Standard Deviation']]
y = combined_df.loc[:, ['Increase']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1, shuffle=False)
print("datasets split")
classifier = svm.SVC(kernel='linear')
start = time.time()
model = classifier.fit(x_train, np.ravel(y_train))
stop = time.time()
print("Classification runtime: " + str(stop - start))
print("training complete")
y_predict = classifier.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_predict))
cf_matrix = confusion_matrix(y_test, y_predict)
auc = metrics.roc_auc_score(y_test, y_predict)
print('The AUC score is: ' + auc)

axis = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
axis.set_title('SVM Confusion Matrix')
axis.set_xlabel('Predicted Values')
axis.set_ylabel('Actual Values')
axis.xaxis.set_ticklabels(['False', 'True'])
axis.yaxis.set_ticklabels(['False', 'True'])
plt.show()


"""combined_df.pop('Company')
combined_df.pop('Increase')
print(combined_df)
model.predict(combined_df)
"""
