import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

df = pd.read_csv('Macrotrends-s-p-500-index-daily.csv', skiprows=8)

work = df[(df['Date'] >= '2020-01-01')].set_index('Date')
work.index = pd.to_datetime(work.index)

fig = px.line(data_frame = work )
fig.write_html('first_figure.html', auto_open=True)

plt.plot_date(x = work.index, y= work)
plt.legend()