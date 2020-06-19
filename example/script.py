import math
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from emm import EMM


if __name__ == '__main__':
    df = pd.read_csv('data/titanic.csv')
    clf = EMM(width=20, depth=1, evaluation_metric='distribution_cosine')
    clf.search(df, target_cols=['Survived'], descriptive_cols=['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                                                               'Fare', 'Cabin', 'Embarked'])
    clf.visualise()

