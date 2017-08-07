import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd

import numpy as np
df_elem = pd.read_csv('elem_school.csv')
df_elem = df_elem.dropna()
df_elem_math = df_elem[df_elem['test_id'] == 'math']
df_elem_ela = df_elem[df_elem['test_id'] == 'ela']

df_middle = pd.read_csv('middle_school.csv')
df_middle = df_middle.dropna()
df_middle_math = df_middle[df_middle['test_id'] == 'math']
df_middle_ela = df_middle[df_middle['test_id'] == 'ela']

df_high = pd.read_csv('high_school.csv')
df_high = df_high.dropna()
df_high_math = df_high[df_high['test_id'] == 'math']
df_high_ela = df_high[df_high['test_id'] == 'ela']

plotly.tools.set_credentials_file(username='jpeng63', api_key='tKN0IRSKAosifJ5w3XnB')

# Create traces
dataset_names = ['elementary schools', 'middle schools', 'high schools']
colors = ['blue', 'red', 'green']
ela_dfs = [df_elem_ela, df_middle_ela, df_high_ela]
math_dfs = [df_elem_math, df_middle_math, df_high_math]
lines = ['-0.573317x+(0.740179)', '-0.576105x+(0.695314)', '-0.461246x+(0.559392)']


data = []

for i in range(3):
    trace = go.Scatter(
        x = ela_dfs[i]['fr_percent'],
        y = ela_dfs[i]['percent_met_and_above'],
        mode = 'markers',
        name = dataset_names[i],
        text = ela_dfs[i]['school_name'],
        marker = dict(color = colors[i]),
        opacity = 0.1,
        showlegend = False,
        hoverinfo = 'none'
    )
    data.append(trace)
    trace = go.Scatter(
        x=ela_dfs[i]['fr_percent'],
        y=ela_dfs[i]['percent_met_and_above'],
        mode='markers',
        marker=dict(color=colors[i], line = dict(width = 1)),
        hoverinfo = "text",
        name=dataset_names[i],
        text=ela_dfs[i]['school_name']
    )
    data.append(trace)

# Plot and embed in ipython notebook!
layout = {"hovermode" : "closest",
          "xaxis": {"fixedrange": True},
          "yaxis": {"fixedrange": True}}

fig = dict(data=data, layout=layout)
py.iplot(fig,  filename='basic-scatter', )

xi = ela_dfs[0]['fr_percent']
trace = go.Scattergl(x=xi,
                  y=-0.573317*xi + (0.740179),
                  mode='lines',
                  marker=go.Marker(color=colors[0]),
                  name='Fit',
                showlegend= 'False',
                     opacity = 0.25
                  )
data.append(trace)

xi = ela_dfs[1]['fr_percent']
trace = go.Scattergl(x=xi,
                  y=-0.576105*xi+(0.695314),
                  mode='lines',
                  marker=go.Marker(color=colors[1]),
                  name='Fit',
                showlegend= 'False',
                     opacity = 0.25
                  )
data.append(trace)

xi = ela_dfs[2]['fr_percent']
trace = go.Scattergl(x=xi,
                  y=-0.461246*xi+(0.559392),
                  mode='lines',
                  marker=go.Marker(color=colors[2]),
                  name='Fit',
                showlegend= 'False',
                     opacity = 0.25
                  )
data.append(trace)

