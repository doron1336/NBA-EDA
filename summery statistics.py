import statsmodels.formula.api as smf
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('./nba-dataset/nba_team_stats_00_to_21.csv', sep=',')

print(df.columns)
print('\n')
print(df.head())
# Match team names

print(df['TEAM'].unique())
print('\n')
print(pd.crosstab(index=df['TEAM'], columns='count'))

for i in range(len(df)):
    if df.loc[i, 'TEAM'] == 'New Jersey Nets':
        df.loc[i, 'TEAM'] = 'Brooklyn Nets'
    elif df.loc[i, 'TEAM'] == 'Charlotte Bobcats':
        df.loc[i, 'TEAM'] = 'Charlotte Hornets'
    elif df.loc[i, 'TEAM'] == 'Los Angeles Clippers':
        df.loc[i, 'TEAM'] = 'LA Clippers'
    elif df.loc[i, 'TEAM'] == 'Vancouver Grizzlies':
        df.loc[i, 'TEAM'] = 'Memphis Grizzlies'
    elif df.loc[i, 'TEAM'] == 'New Orleans Hornets' or df.loc[i, 'TEAM'] == 'New Orleans/Oklahoma City Hornets':
        df.loc[i, 'TEAM'] = 'New Orleans Pelicans'
    elif df.loc[i, 'TEAM'] == 'Seattle SuperSonics':
        df.loc[i, 'TEAM'] = 'Oklahoma City Thunder'

df = df.sort_values(by='SEASON')
# print(df.head())
# print(df[['REB', 'AST', '3PM', 'FGM', 'STL', 'BLK']].describe())
df[['REB', 'AST', '3PM', 'FGM', 'STL', 'BLK']
   ].describe().to_csv('summery_stat.csv')
