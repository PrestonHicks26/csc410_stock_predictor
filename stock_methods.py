def _produce_prediction(data, window):
    '''
    Function that produces 'truth; values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :param window: number of days, or rows to look ahead to see what the price did
    '''

    prediction = (data.shift(-window)['close'] >= data['close'])
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)

    return data

  
def _exponential_smooth(data, alpha):
  
  return data.ewm(alpha=alpha).mean()

# also have a prediction method but im not sure its usuable with our dataset so lmk if anyone wants it posted
    
