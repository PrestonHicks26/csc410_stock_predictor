import pandas as pd
import numpy as np

# uses collection of data from supermarkets and departments, according to R. Choudhry and K. Garg stock prices within
# same industry are highly correlated

# remove close and keep adjusted close as, according to investopedia, adjusted close takes into account additional
# factors and is more useful for analyzing historical data

big_lots_df = pd.read_csv('BIG.csv')
costco_df =pd.read_csv('COST.csv')
ingles_markets_df =pd.read_csv('IMKTA.csv')
kroger_df =pd.read_csv('KR.csv')
spartannash_df =pd.read_csv('SPTN.csv')
target_df =pd.read_csv('TGT.csv')
village_super_market_pd =pd.read_csv('VLGEA.csv')
weis_markets_df =pd.read_csv('WMK.csv')
walmart_df =pd.read_csv('WMT.csv')

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
    o_min_c = []
    _7_day_ma = []
    _14_day_ma = []
    _21_day_ma = []
    _7_days_std_dev = []
    for i in range(df.shape[0]):
        company.append(name)
        h_min_l.append(df.at[i, 'High'] - df.at[i, 'Low'])
        o_min_c.append(df.at[i, 'Open'] - df.at[i, 'Close'])
        _7_day_ma.append(moving_average(i, 7, df))
        _14_day_ma.append(moving_average(i, 14, df))
        _21_day_ma.append(moving_average(i, 21, df))
        if i < df.shape[0]-7:
            _7_days_std_dev.append(np.std(df.iloc[i:i+7, 6]))
        else:
            _7_days_std_dev.append(np.std(df.iloc[i:, 6]))
    df.drop('Close', axis=1, inplace=True)
    df.insert(0, 'Company', company)
    df['H-L'] = h_min_l
    df['O-C'] = o_min_c
    df['7 Day MA'] = _7_day_ma
    df['14 Day MA'] = _14_day_ma
    df['21 Day MA'] = _21_day_ma
    df['7 Days Standard Deviation'] = _7_days_std_dev


dataFrames = {'Big Lots': big_lots_df, 'Cost Co.': costco_df, 'Ingles Markets': ingles_markets_df, 'Kroger': kroger_df,
              'SpartanNash': spartannash_df, 'Target': target_df, 'Village Super Market': village_super_market_pd,
              'Weis Markets': weis_markets_df, 'Walmart': walmart_df}

for company in dataFrames:
    column_modification(company, dataFrames[company])

combined = pd.concat([big_lots_df, costco_df, ingles_markets_df, kroger_df, spartannash_df, target_df,
                     village_super_market_pd, weis_markets_df, walmart_df])
combined.to_csv('combined.csv')