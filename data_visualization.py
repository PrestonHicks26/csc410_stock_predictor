import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

test_df = pd.read_csv('test.csv')
#boxplot = sns.boxplot(y='Adj Close', x='Company', data=test_df)
features = ['Open', 'High', 'Low', 'Volume', 'H-L', 'O-AdjC', '7 Day MA', '14 Day MA', '21 Day MA',
            '7 Days Standard Deviation']
for feature in features:
    test_df.plot.scatter(x=feature, y='Adj Close')
plt.show()