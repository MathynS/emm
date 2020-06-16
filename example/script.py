import math
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from EMM import EMM


if __name__ == '__main__':
    df = pd.read_csv('titanic.csv')
    clf = EMM(width=20, depth=3, evaluation_metric='distribution_cosine')
    clf.search(df, target_cols=['Survived'], descriptive_cols=['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                                                               'Fare', 'Cabin', 'Embarked'])

    fig = make_subplots(rows=6, cols=2, subplot_titles=['all'] + [str(s.description) for s in clf.beam.subgroups])
    values = df['Survived'].value_counts()
    fig.add_trace(
        go.Bar(x=values.index, y=values.values),
        row=1, col=1
    )
    for i, subgroup in enumerate(clf.beam.subgroups[:10]):
        fig.add_trace(
            go.Bar(x=subgroup.target.index, y=subgroup.target.values),
            row=math.floor((i+1) / 2) + 1, col=((i+1) % 2) + 1
        )
    fig.show()
