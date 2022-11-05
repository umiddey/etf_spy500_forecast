import datetime

from polygon import RESTClient
import pandas as pd
from typing import cast
from urllib3 import HTTPResponse
import json
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import time
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA


# Making the daterange df to use for iteration
date_ranges = pd.DataFrame(pd.date_range('2021-01-01', 'now', freq='7D'), columns=['start'])
date_ranges['end'] = date_ranges['start'].shift(-1) # last row has a null in end column, so needs to be removed

# Choosing all except last row
date_ranges['end'].fillna(pd.to_datetime('now'), inplace = True)
date_ranges = date_ranges.iloc[:-1,:]
df = pd.DataFrame()

client = RESTClient('xMEnZ6YR2t7ekLzw3UxkDzcTb5IzLk9I')
for i in range(0, len(date_ranges)):

    time.sleep(15)

    try:
        aggs = cast(
            HTTPResponse,
            client.get_aggs(
                "SPY",
                15,
                "minute",
                f"{date_ranges.iloc[i]['start'].date()}",
                f"{date_ranges.iloc[i]['end'].date()}",
                raw=True,
                sort= 'asc'
            ),
        )

        raw_data = aggs.data
        jsonized_data = json.loads(raw_data)

        temp = pd.DataFrame.from_dict(jsonized_data['results'])

        temp['dt'] = pd.DatetimeIndex( pd.to_datetime(temp['t'],unit='ms') )

        df = df.append(temp)
    except: continue


df.to_csv('spy500_data_2021_2022.csv')

df = pd.read_csv('spy500_data_2021_2022.csv', index_col= 'dt')

work = df[df.index >= "2022-05-01"]
work.index = pd.to_datetime(work.index)


dt_range = pd.date_range('2022-10-29', '2022-11-15', freq= '900s')

ind = (dt_range.indexer_between_time('00:30','08:45').tolist())
# print (ind)

dt_range = pd.DataFrame(dt_range[np.isin(np.arange(len(dt_range)), ind, invert=True)],
                        columns=['dt'])

work_df = pd.merge(work.reset_index(), dt_range, on = 'dt', how='outer')
work_df.set_index('dt', inplace=True)
work_df.index = pd.to_datetime(work_df.index)
df_inter = work_df.interpolate(method='polynomial', order = 2)


fig = px.line(data_frame=df_inter,x=df_inter.index, y=['o'])
fig.write_html('first_figure.html', auto_open=True)

fig2 = px.line(data_frame=work, x = work.index, y=['o'])
fig2.write_html('first_figure2.html', auto_open=True)

# statsmodel

y = work[work.index < "2022-09-30"]['o']
model = SARIMAX(y, order = (5, 4, 2), seasonal_order=(2,2,2,12))
model = model.fit()

test = work[work.index >= "2022-09-30"]['o']
y_pred = model.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)
y_pred_df["Predictions"] = model.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y.append(y_pred_df["Predictions"])
plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.legend()


ARIMAmodel = ARIMA(y, order = (2, 2, 2))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)
y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y.append(y_pred_df["Predictions"] )
plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
plt.legend()



fig = plt.figure()
ax1 = fig.add_axes((0.1,0.4,0.5,0.5))

ax1.set_title('Title of Plot')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax1.plot(temp['o'], temp['c'],c='black',linestyle='--',linewidth=0.5)