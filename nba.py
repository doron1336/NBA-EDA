import pandas as pd
import numpy as np
from math import pi
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from IPython.display import display, Markdown

games_details = pd.read_csv('./nba-dataset/games_details.csv')
players = pd.read_csv('./nba-dataset/players.csv')
teams = pd.read_csv('./nba-dataset/teams.csv')
ranking = pd.read_csv('./nba-dataset/ranking.csv')
games = pd.read_csv('./nba-dataset/games.csv')


def print_missing_values(df):
    df_null = pd.DataFrame(len(df) - df.notnull().sum(), columns=['Count'])
    df_null = df_null[df_null['Count'] > 0].sort_values(
        by='Count', ascending=False)
    df_null = df_null/len(df)*100

    if len(df_null) == 0:
        display(Markdown('No missing value.'))
        return

    x = df_null.index.values
    height = [e[0] for e in df_null.values]

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x, height, width=0.8)
    plt.xticks(x, x, rotation=60)
    plt.xlabel('Columns')
    plt.ylabel('Percentage')
    plt.title('Percentage of missing values in columns')
    plt.show()


def dataset_overview(df, df_name):
    display(Markdown(f'### {df_name} dataset overview'))
    display(Markdown(f'dataset shape : {df.shape}'))
    display(Markdown(f'#### Display 5 first rows'))
    display(df.head())
    display(Markdown('*****'))
    display(Markdown(f'#### Describe dataset'))
    display(df.describe().T)
    display(Markdown('*****'))
    display(Markdown(f'#### Missing values'))
    print_missing_values(df)


# dataset_overview(games_details, 'games_details')
def convert_min(x):
    if pd.isna(x):
        return 0
    x = str(x).split(':')
    if len(x) < 2:
        rounded_val = round(float(x[0]), 0)
        return int(rounded_val)
    else:
        rounded_val = round(float(x[0]), 0)
        rounded_val1 = round(float(x[1]), 0)
        return int(rounded_val)*60+int(rounded_val1)


def plot_top(df, column, label_col=None, max_plot=5):
    top_df = df.sort_values(column, ascending=False).head(max_plot)

    height = top_df[column]
    x = top_df.index if label_col == None else top_df[label_col]

    gold, silver, bronze, other = ('#FFA400', '#bdc3c7', '#cd7f32', '#3498db')
    colors = [gold if i == 0 else silver if i == 1 else bronze if i ==
              2 else other for i in range(0, len(top_df))]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(x, height, color=colors)
    plt.xticks(x, x, rotation=60)
    plt.xlabel(label_col)
    plt.ylabel(column)
    plt.title(f'Top {max_plot} of {column}')
    plt.show()

# Advanced EDA : let's answer some questions


# players vs number of games in the league
players_name = games_details['PLAYER_NAME']
val_cnt = players_name.value_counts().to_frame().reset_index()
val_cnt.columns = ['PLAYER_NAME', 'Number of games']

plot_top(val_cnt, column='Number of games',
         label_col='PLAYER_NAME', max_plot=10)

# minutes played by players
df_tmp = games_details[['PLAYER_NAME', 'MIN']]
df_tmp.loc[:, 'MIN'] = df_tmp['MIN'].apply(convert_min)
agg = df_tmp.groupby('PLAYER_NAME').agg('sum').reset_index()
agg.columns = ['PLAYER_NAME', 'Number of seconds played']

plot_top(agg, column='Number of seconds played',
         label_col='PLAYER_NAME', max_plot=10)

stats_cols = {
    'FGM': 'Field Goals Made',
    'FGA': 'Field Goals Attempted',
    'FG_PCT': 'Field Goal Percentage',
    'FG3M': 'Three Pointers Made',
    'FG3A': 'Three Pointers Attempted',
    'FG3_PCT': 'Three Point Percentage',
    'FTM': 'Free Throws Made',
    'FTA': 'Free Throws Attempted',
    'FT_PCT': 'Free Throw Percentage',
    'OREB': 'Offensive Rebounds',
    'DREB': 'Defensive Rebounds',
    'REB': 'Rebounds',
    'AST': 'Assists',
    'TO': 'Turnovers',
    'STL': 'Steals',
    'BLK': 'Blocked Shots',
    'PF': 'Personal Foul',
    'PTS': 'Points',
    'PLUS_MINUS': 'Plus-Minus'
}


def agg_on_columns(df, agg_var, operation=['mean']):
    return df[agg_var].agg(operation)


# Remove players that didn't played at a game
df_tmp = games_details[~games_details['MIN'].isna()]
del df_tmp['MIN']

# Define key statistics columns, one for percentage variable and one for other important statistics
prct_var = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
other_var = ['REB', 'AST', 'STL', 'PF', 'BLK']

# Create a specific dataset for LeBron James
lebron_james_df = df_tmp[df_tmp['PLAYER_NAME'] == 'LeBron James']

overall_agg_prct = agg_on_columns(
    df=df_tmp, agg_var=prct_var, operation=['mean'])
overall_agg_other = agg_on_columns(
    df=df_tmp, agg_var=other_var, operation=['mean'])

lebron_james_stats_prct = agg_on_columns(
    df=lebron_james_df, agg_var=prct_var, operation=['mean'])
lebron_james_stats_other = agg_on_columns(
    df=lebron_james_df, agg_var=other_var, operation=['mean'])

stats_prct = pd.concat([lebron_james_stats_prct, overall_agg_prct])
stats_other = pd.concat([lebron_james_stats_other, overall_agg_other])

stats_prct.index = ['Lebron James', 'overall stats']
stats_other.index = ['Lebron James', 'overall stats']
display(Markdown('#### Stats comparison between Lebron James and overall statistics'))
print(stats_prct)
row_list_lebron = stats_prct.loc["Lebron James", :].values.flatten().tolist()
row_list_overall = stats_prct.loc["overall stats", :].values.flatten().tolist()
categories = list(stats_prct)

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=row_list_lebron,
    theta=categories,
    fill='toself',
    name='Lebron James'
))
fig.add_trace(go.Scatterpolar(
    r=row_list_overall,
    theta=categories,
    fill='toself',
    name='overall stats'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True
)

fig.show()

df = games.sort_values(by='SEASON')
print(stats_other)

row_list_lebron_stats_other = stats_other.loc["Lebron James", :].values.flatten(
).tolist()
row_list_overall_stats_other = stats_other.loc["overall stats", :].values.flatten(
).tolist()
categories = list(stats_other)

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=row_list_lebron_stats_other,
    theta=categories,
    fill='toself',
    name='Lebron James'
))
fig.add_trace(go.Scatterpolar(
    r=row_list_overall_stats_other,
    theta=categories,
    fill='toself',
    name='overall stats'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 8]
        )),
    showlegend=True
)

fig.show()
