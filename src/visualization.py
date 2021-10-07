import math
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import plotly.express as px

import subgroup


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
        annotations.extend([
            {'x': -0.07,
             'y': 0.5,
             'showarrow': False,
             'text': target_columns[1],
             'textangle': -90,
             'xref': "paper",
             'yref': "paper"}
        ])

    return annotations


def show_plots(fig, group_size, cols, target_columns, height):
    """Shows the plots with a given subplot height."""
    fig.update_layout(
        height=height * math.ceil((group_size + 1) / cols),
        showlegend=False,
        xaxis_title=target_columns[0]
    )
    if len(target_columns) < 1:
        fig.update_layout(yaxis_title=target_columns[1])
    if len(target_columns) < 2:
        fig.update_layout(zaxis_title=target_columns[2])

    # fig["layout"]["annotations"] = default_annotations(fig, target_columns)
    fig.show()


def heatmap(dataset: subgroup.Subgroup,
            subgroups: list[subgroup.Subgroup],
            target_columns: list[str],
            translations: dict,
            cols: int,
            group_size: int,
            include_dataset: bool,
            height: int = 450):
    """Visualizer for the covariance quality measure. Shows a heatmap.

    Args:
        dataset: The entire dataset.
        subgroups: List of subgroups that should be visualized.
        target_columns: Name sof the target columns.
        translations: Translations that should be done.
        cols: Number of columns of plots.
        group_size: Size of groupings.
        include_dataset: Whether or not to include the entire dataset in the
            visualization.
        height: Height of each plot. Defaults to 450 px.
    """
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
    show_plots(fig, group_size, cols, target_columns, height)


def correlation(dataset: subgroup.Subgroup,
                subgroups: list[subgroup.Subgroup],
                target_columns: list[str],
                translations: dict,
                cols: int,
                group_size: int,
                include_dataset: bool,
                height: int = 450):
    """Visualizer for the covariance quality measure. Shows a scatter plot.

    Args:
        dataset: The entire dataset.
        subgroups: List of subgroups that should be visualized.
        target_columns: Name sof the target columns.
        translations: Translations that should be done.
        cols: Number of columns of plots.
        group_size: Size of groupings.
        include_dataset: Whether or not to include the entire dataset in the
            visualization.
        height: Height of each plot. Defaults to 450 px.
    """
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
    show_plots(fig, group_size, cols, target_columns, height)


def distribution(dataset: subgroup.Subgroup,
                 subgroups: list[subgroup.Subgroup],
                 target_columns: list[str],
                 translations: dict,
                 cols: int,
                 group_size: int,
                 include_dataset: bool,
                 height: int = 450):
    """Visualizer for the distribution quality measure. Shows column charts.

        Args:
            dataset: The entire dataset.
            subgroups: List of subgroups that should be visualized.
            target_columns: Name sof the target columns.
            translations: Translations that should be done.
            cols: Number of columns of plots.
            group_size: Size of groupings.
            include_dataset: Whether or not to include the entire dataset in the
                visualization.
            height: Height of each plot. Defaults to 450 px.
        """
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
    show_plots(fig, group_size, cols, target_columns, height)


def covariance(dataset: subgroup.Subgroup,
               subgroups: list[subgroup.Subgroup],
               target_columns: list[str],
               translations: dict,
               cols: int,
               group_size: int,
               include_dataset: bool,
               height: int = 450):
    """Visualizer for the covariance measure. Shows a 2D or 3D scatter plot.

    Args:
        dataset: The entire dataset.
        subgroups: List of subgroups that should be visualized.
        target_columns: Name sof the target columns.
        translations: Translations that should be done.
        cols: Number of columns of plots.
        group_size: Size of groupings.
        include_dataset: Whether or not to include the entire dataset in the
            visualization.
        height: Height of each plot. Defaults to 450 px.
    """
    if len(target_columns) < 3:
        # If there are only 2 target columns, then the use the correlation
        # visualizer instead.
        correlation(dataset, subgroups, target_columns, translations, cols,
                    group_size, include_dataset, height)
        return

    sets, titles = setup_sets_titles(include_dataset, dataset, subgroups,
                                     group_size)

    fig = make_subplots(rows=math.ceil((group_size + 1) / cols),
                        cols=cols,
                        subplot_titles=titles,
                        vertical_spacing=0.2)

    for i, subgroup in enumerate(sets):
        if len(target_columns) == 3:
            marker_style = {'size': 4,
                            'opacity': 0.8}
            fig.add_trace(go.Scatter3d(x=subgroup.data[target_columns[0]],
                                       y=subgroup.data[target_columns[1]],
                                       z=subgroup.data[target_columns[2]],
                                       mode='markers',
                                       marker=marker_style))
        elif len(target_columns) == 4:
            marker_style = {'size': 4,
                            'color': subgroup.data[target_columns[3]],
                            'colorscale': 'Viridis',
                            'opacity': 0.8}
            fig.add_trace(go.Scatter3d(x=subgroup.data[target_columns[0]],
                                       y=subgroup.data[target_columns[1]],
                                       z=subgroup.data[target_columns[2]],
                                       mode='markers',
                                       marker=marker_style))
        else:
            raise UserWarning("Cannot visualize more than 4 target columns")

        show_plots(fig, group_size, cols, target_columns, height)


visualizations = {'correlation': correlation,
                  'distribution_cosine': distribution,
                  'regression': correlation,
                  'WRAcc': distribution,
                  'covariance': covariance,
                  'heatmap': heatmap}
