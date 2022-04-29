import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def moving_average(i, window, df):
    if i<df.shape[0]-window:
        # selects from adj close column
        selection = df.iloc[i:i+window,6]
        sum = 0
        for num in selection:
            sum += num
        return sum/window
    else: # handles values where window goes past end of column, unsure if this is correct solution
        selection = df.iloc[i:, 6]
        sum = 0
        for num in selection:
            sum += num
        return sum / (df.shape[0]-i)


# ----------------------------------------------------------------------------------------------------------------------
sp500_df = pd.read_csv('sp500_stocks.csv')

print(sp500_df.head())

# adjust range to be same as primary dataset
temp_df = pd.DataFrame()
for i in range(505):
    temp_df = pd.concat([temp_df, sp500_df.iloc[i*3080+1:2523+3080*i, :]])
sp500_df = temp_df
sp500_df.to_csv('range_test.csv')
sp500_df = pd.read_csv('range_test.csv') # concat messed up indices, by saving and reloading, pd resets indices
sp500_df.drop(sp500_df.columns[0], axis=1, inplace=True)
print(sp500_df.head())

# rearrange columns
sp500_df.drop('Close', axis=1, inplace=True)
company = sp500_df.pop('Symbol')
sp500_df.insert(0, 'Company', company)
open = sp500_df.pop('Open')
sp500_df.insert(2, 'Open', open)
adj_close = sp500_df.pop('Adj Close')
sp500_df.insert(5, 'Adj Close', adj_close)

#same as column modification method but without company column
h_min_l = []
o_min_adjc = []
_7_day_ma = []
_14_day_ma = []
_21_day_ma = []
_7_days_std_dev = []
increase = []
for i in range(sp500_df.shape[0]):
    h_min_l.append(sp500_df.at[i, 'High'] - sp500_df.at[i, 'Low'])
    o_min_adjc.append(sp500_df.at[i, 'Open'] - sp500_df.at[i, 'Adj Close'])
    _7_day_ma.append(moving_average(i, 7, sp500_df))
    _14_day_ma.append(moving_average(i, 14, sp500_df))
    _21_day_ma.append(moving_average(i, 21, sp500_df))
    if i < sp500_df.shape[0]-7:
        _7_days_std_dev.append(np.std(sp500_df.iloc[i:i+7, 6]))
    else:
        _7_days_std_dev.append(np.std(sp500_df.iloc[i:, 6]))
    if i < sp500_df.shape[0]-1:
        """if (sp500_df.at[i + 1, 'Adj Close'] - sp500_df.at[i, 'Adj Close']) == 0: # checks how often values are equal to see if alt solution necessary
            print('Equal')
            print(sp500_df.at[i, 'Date'])
            print(sp500_df.at[i + 1, 'Adj Close'])
            print(sp500_df.at[i, 'Adj Close'])"""
        if (sp500_df.at[i+1, 'Adj Close']-sp500_df.at[i, 'Adj Close'])>0:
            increase.append(True)
        else:
            increase.append(False)
    else:
        increase.append(False)

sp500_df['H-L'] = h_min_l
sp500_df['O-AdjC'] = o_min_adjc
sp500_df['7 Day MA'] = _7_day_ma
sp500_df['14 Day MA'] = _14_day_ma
sp500_df['21 Day MA'] = _21_day_ma
sp500_df['7 Days Standard Deviation'] = _7_days_std_dev
sp500_df['Increase'] = increase

sp500_df.to_csv('sp500prepped.csv')
train, test = train_test_split(sp500_df, test_size=.3, random_state=1)
train.to_csv('sp500train.csv')
test.to_csv('sp500test.csv')
