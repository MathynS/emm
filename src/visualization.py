import math
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def heatmap(dataset, subgroups, target_columns, translations, cols, group_size, include_dataset):
    sets = [dataset] + subgroups[:group_size] if include_dataset else subgroups[:group_size]
    titles = ['Dataset'] + [str(s.description) for s in subgroups[:group_size]] if include_dataset else \
        [str(s.description) for s in subgroups[:group_size]]

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=titles,
                        vertical_spacing=0.39)

    for i, subgroup in enumerate(sets):
        fig.add_trace(
            go.Heatmap(z=subgroup.target, x=translations[target_columns[1]], y=translations[target_columns[0]],
                       legendgroup=str(i)),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )

    # fig.update_xaxes(range=[dataset.data[target_columns[0]].min(), dataset.data[target_columns[0]].max()])
    # fig.update_yaxes(range=[dataset.data[target_columns[1]].min(), dataset.data[target_columns[1]].max()])
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


def correlation(dataset, subgroups, target_columns, translations, cols, group_size, include_dataset):
    sets = [dataset] + subgroups[:group_size] if include_dataset else subgroups[:group_size]
    titles = ['Dataset'] + [str(s.description) for s in subgroups[:group_size]] if include_dataset else \
        [str(s.description) for s in subgroups[:group_size]]

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=titles,
                        vertical_spacing=0.2)

    for i, subgroup in enumerate(sets):
        fig.add_trace(
            go.Scatter(x=subgroup.data[target_columns[0]], y=subgroup.data[target_columns[1]], mode='markers'),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
        fig.add_trace(
            go.Scatter(x=subgroup.data[target_columns[0]], y=(subgroup.data[target_columns[0]] * subgroup.target),
                       mode='lines'),
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


def distribution(dataset, subgroups, target_columns, translations, cols, group_size, include_dataset):
    sets = [dataset] + subgroups[:group_size] if include_dataset else subgroups[:group_size]
    titles = ['Dataset'] + [str(s.description) for s in subgroups[:group_size]] if include_dataset else \
        [str(s.description) for s in subgroups[:group_size]]

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=titles,
                        vertical_spacing=0.3)

    for i, subgroup in enumerate(sets):
        X = [translations[target_columns[0]][x] for x in subgroup.target.index]
        fig.add_trace(
            go.Bar(x=X, y=subgroup.target.values),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
    # fig.update_yaxes(range=[0, dataset.target.values.max()])
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
    regression=correlation,
    WRAcc=distribution,
    heatmap=heatmap
)
