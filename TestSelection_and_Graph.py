import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv('sp500_stocks.csv')

df.head()
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
# combining multiple data points into the same graph
fig.update_layout(xaxis_title='Date', yaxis_title='Price', title='User Select Sample plot')
fig.show()
# interactive visualization
