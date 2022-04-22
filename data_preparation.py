import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# uses collection of data from supermarkets and departments, according to R. Choudhry and K. Garg stock prices within
# same industry are highly correlated

# remove close and keep adjusted close as, according to investopedia, adjusted close takes into account additional
# factors and is more useful for analyzing historical data

big_df = pd.read_csv('BIG.csv')
cost_df =pd.read_csv('COST.csv')
imkta_df =pd.read_csv('IMKTA.csv')
kr_df =pd.read_csv('KR.csv')
sptn_df =pd.read_csv('SPTN.csv')
tgt_df =pd.read_csv('TGT.csv')
vlgea_df =pd.read_csv('VLGEA.csv')
wmk_df =pd.read_csv('WMK.csv')
wmt_df =pd.read_csv('WMT.csv')

#add company name column and additional variables
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

def column_modification(name, df):
    company = []
    h_min_l = []
    o_min_adjc = []
    _7_day_ma = []
    _14_day_ma = []
    _21_day_ma = []
    _7_days_std_dev = []
    increase = []
    for i in range(df.shape[0]):
        company.append(name)
        h_min_l.append(df.at[i, 'High'] - df.at[i, 'Low'])
        o_min_adjc.append(df.at[i, 'Open'] - df.at[i, 'Adj Close'])
        _7_day_ma.append(moving_average(i, 7, df))
        _14_day_ma.append(moving_average(i, 14, df))
        _21_day_ma.append(moving_average(i, 21, df))
        if i < df.shape[0]-7:
            _7_days_std_dev.append(np.std(df.iloc[i:i+7, 6]))
        else:
            _7_days_std_dev.append(np.std(df.iloc[i:, 6]))
        if i < df.shape[0]-1:
            if (df.at[i + 1, 'Adj Close'] - df.at[i, 'Adj Close']) == 0: # checks how often values are equal to see if alt solution necessary
                print('Equal')
                print(df.at[i, 'Date'])
                print(df.at[i + 1, 'Adj Close'])
                print(df.at[i, 'Adj Close'])
            if (df.at[i+1, 'Adj Close']-df.at[i, 'Adj Close'])>0:
                increase.append(True)
            else:
                increase.append(False)
        else:
            increase.append(False)

    df.drop('Close', axis=1, inplace=True)
    df.insert(0, 'Company', company)
    df['H-L'] = h_min_l
    df['O-AdjC'] = o_min_adjc
    df['7 Day MA'] = _7_day_ma
    df['14 Day MA'] = _14_day_ma
    df['21 Day MA'] = _21_day_ma
    df['7 Days Standard Deviation'] = _7_days_std_dev
    df['Increase'] = increase


dataFrames = {'BIG': big_df, 'COST': cost_df, 'IMKTA': imkta_df, 'KR': kr_df,
              'SPTN': sptn_df, 'TGT': tgt_df, 'VLGEA': vlgea_df,
              'WMK': wmk_df, 'WMT': wmt_df}

for company in dataFrames:
    column_modification(company, dataFrames[company])

combined = pd.concat([big_df, cost_df, imkta_df, kr_df, sptn_df, tgt_df,
                      vlgea_df, wmk_df, wmt_df])
combined.to_csv('combined.csv')

train, test = train_test_split(combined, test_size=.3, random_state=1)
train.to_csv('train.csv')
test.to_csv('test.csv')

