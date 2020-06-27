import math
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression


def correlation(dataset, subgroups, target_columns, cols: int = 3, group_size: int = 9):

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=['all'] + [str(s.description) for s in subgroups[:group_size]])

    for i, subgroup in enumerate([dataset] + subgroups[:group_size]):
        reg = LinearRegression().fit(np.vstack(subgroup.data[target_columns[0]]), subgroup.data[target_columns[1]])
        subgroup.data['bestfit'] = reg.predict(np.vstack(subgroup.data[target_columns[0]]))
        fig.add_trace(
            go.Scatter(x=subgroup.data[target_columns[0]], y=subgroup.data[target_columns[1]], mode='markers'),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
        fig.add_trace(
            go.Scatter(x=subgroup.data[target_columns[0]], y=subgroup.data['bestfit'], mode='lines'),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
    fig.update_xaxes(range=[dataset.data[target_columns[0]].min(), dataset.data[target_columns[0]].max()])
    fig.update_yaxes(range=[dataset.data[target_columns[1]].min(), dataset.data[target_columns[1]].max()])
    fig.update_layout(
        showlegend=False
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


def distribution(dataset, subgroups, target_columns, cols: int = 3, group_size: int = 9):

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=['all'] + [str(s.description) for s in subgroups[:group_size]])

    for i, subgroup in enumerate([dataset] + subgroups[:group_size]):
        fig.add_trace(
            go.Bar(x=subgroup.target.index, y=subgroup.target.values),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
    fig.update_yaxes(range=[0, dataset.target.values.max()])
    fig.update_layout(
        showlegend=False
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
            text='Frequency count',
            textangle=-90,
            xref="paper",
            yref="paper"
        )
    ])
    fig["layout"]["annotations"] = annotations
    fig.show()


visualizations = dict(
    correlation=correlation,
    distribution_cosine=distribution,
    regression=correlation
)
