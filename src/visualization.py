import math
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def setup_sets_titles(include_dataset, dataset, subgroups, group_size):
    """Sets the sets and titles required by visualizer functions.

    Returns: (tuple) the first element is the sets, the second element is the
        titles.
    """
    if include_dataset:
        sets = [dataset] + subgroups[:group_size]
        titles = ['Dataset'] + [str(s.description)
                                for s in subgroups[:group_size]]
    else:
        sets = subgroups[:group_size]
        titles = [str(s.description) for s in subgroups[:group_size]]

    return sets, titles


def default_annotations(fig, target_columns):
    """Provides default annotations for visualizations."""
    annotations = [a.to_plotly_json() for a in fig["layout"]["annotations"]]
    annotations.extend([
        {"x": 0.5,
         "y": -0.15,
         "showarrow": False,
         "text": target_columns[0],
         "xref": "paper",
         "yref": "paper"},
    ])
    if len(target_columns) > 1:
        for i in range(len(target_columns) - 1):
            annotations.extend([
                {'x': -0.07,
                 'y': 0.5,
                 'showarrow': False,
                 'text': target_columns[i + 1],
                 'textangle': -90,
                 'xref': "paper",
                 'yref': "paper"}
            ])
    return annotations


def heatmap(dataset, subgroups, target_columns, translations, cols, group_size,
            include_dataset: bool):
    sets, titles = setup_sets_titles(include_dataset, dataset, subgroups,
                                     group_size)

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=titles,
                        vertical_spacing=0.39)

    for i, subgroup in enumerate(sets):
        fig.add_trace(
            go.Heatmap(z=subgroup.target, x=translations[target_columns[1]],
                       y=translations[target_columns[0]],
                       legendgroup=str(i)),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )

    # fig.update_xaxes(range=[dataset.data[target_columns[0]].min(),
    #                         dataset.data[target_columns[0]].max()])
    # fig.update_yaxes(range=[dataset.data[target_columns[1]].min(),
    #                         dataset.data[target_columns[1]].max()])
    fig.update_layout(
        showlegend=False
    )
    fig["layout"]["annotations"] = default_annotations(fig, target_columns)
    fig.show()


def correlation(dataset, subgroups, target_columns, translations, cols,
                group_size, include_dataset):
    sets, titles = setup_sets_titles(include_dataset, dataset, subgroups,
                                     group_size)

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=titles,
                        vertical_spacing=0.2)

    for i, subgroup in enumerate(sets):
        fig.add_trace(
            go.Scatter(x=subgroup.data[target_columns[0]],
                       y=subgroup.data[target_columns[1]], mode='markers'),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
        fig.add_trace(
            go.Scatter(x=subgroup.data[target_columns[0]],
                       y=(subgroup.data[target_columns[0]] * subgroup.target),
                       mode='lines'),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
    fig.update_xaxes(range=[dataset.data[target_columns[0]].min(),
                            dataset.data[target_columns[0]].max()])
    fig.update_yaxes(range=[dataset.data[target_columns[1]].min(),
                            dataset.data[target_columns[1]].max()])
    fig.update_layout(
        showlegend=False
    )

    fig["layout"]["annotations"] = default_annotations(fig, target_columns)
    fig.show()


def distribution(dataset, subgroups, target_columns, translations, cols,
                 group_size, include_dataset):
    sets, titles = setup_sets_titles(include_dataset, dataset, subgroups,
                                     group_size)

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols), cols=cols,
                        subplot_titles=titles,
                        vertical_spacing=0.3)

    for i, subgroup in enumerate(sets):
        # X = [target_columns[0][x] for x in subgroup.target.index]
        fig.add_trace(
            go.Bar(x=subgroup.data[target_columns[0]],
                   y=subgroup.target.values),
            row=math.floor(i / cols) + 1, col=(i % cols) + 1
        )
    # fig.update_yaxes(range=[0, dataset.target.values.max()])
    fig.update_layout(
        showlegend=False
    )

    fig["layout"]["annotations"] = default_annotations(fig, target_columns)
    fig.show()


visualizations = {'correlation': correlation,
                  'distribution_cosine': distribution,
                  'regression': correlation, 'WRAcc': distribution,
                  'heatmap': heatmap}
