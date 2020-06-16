import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

from EMM import EMM
from subgroup import Subgroup
from description import Description


if __name__ == '__main__':
    df = pd.read_csv('Housing.csv', index_col=0)
    target_columns = ['lotsize', 'price']
    clf = EMM(width=40, depth=3, evaluation_metric='correlation')
    clf.search(df, target_cols=target_columns)

    cols = 3
    group_size = 8
    dataset = Subgroup(df, Description('all'))

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=['all'] + [str(s.description) for s in clf.beam.subgroups[:group_size]])

    for i, subgroup in enumerate([dataset] + clf.beam.subgroups[:group_size]):
        reg = LinearRegression().fit(np.vstack(subgroup.data[target_columns[0]]), subgroup.data['price'])
        subgroup.data['bestfit'] = reg.predict(np.vstack(subgroup.data[target_columns[0]]))
        fig.add_trace(
            go.Scatter(x=subgroup.data[target_columns[0]], y=subgroup.data['price'], mode='markers'),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
        fig.add_trace(
            go.Scatter(x=subgroup.data[target_columns[0]], y=subgroup.data['bestfit'], mode='lines'),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
    fig.update_xaxes(range=[dataset.data[target_columns[0]].min(), dataset.data[target_columns[0]].max()])
    fig.update_yaxes(range=[dataset.data[target_columns[1]].min(), dataset.data[target_columns[1]].max()])
    fig.update_layout(
        showlegend=False,
        width=1200,
    )
    annotations = [a.to_plotly_json() for a in fig["layout"]["annotations"]]
    annotations.extend([
        dict(
            x=0.5,
            y=-0.15,
            showarrow=False,
            text=target_columns[0],
            xref="paper",
            yref="paper"
        ),
        dict(
            x=-0.07,
            y=0.5,
            showarrow=False,
            text=target_columns[1],
            textangle=-90,
            xref="paper",
            yref="paper"
        )
    ])
    fig["layout"]["annotations"] = annotations
    fig.show()