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
df.loc[:, 'WIN%'] = df.loc[:, 'WIN%'] * 100
sns.regplot(data=df, x='3P%', y='WIN%', line_kws={'color': 'orange'})
sns.set(rc={"figure.figsize": (12, 6)})
plt.title('Distribution and Correlation between WIN% and 3P%')
plt.style.use('seaborn')
plt.savefig('win_3p_scatter.png')

avg_3pp_ssn = df.groupby('SEASON')['3P%'].mean()
avg_3pa_ssn = df.groupby('SEASON')['3PA'].mean()
avg_3pm_ssn = df.groupby('SEASON')['3PM'].mean()

avg_3p_data_ssn = pd.concat([avg_3pp_ssn, avg_3pa_ssn, avg_3pm_ssn], axis=1)
avg_3p_data_ssn.to_csv('avg_3p_data_ssn.csv')
print(avg_3p_data_ssn)
print('\n')

# avg_3p_data_ssn.plot(kind='line', figsize=(16, 5))
# plt.title('Evolution of "3PA", "3PM" and "3P%" over the Course of 20 NBA Seasons')
# plt.style.use('ggplot')
# plt.savefig('3p_stats_over_time.png')

# # Average "3P%"", "3PA", "3PM" in Season 2000/2001

# print('Average "3P%"", "3PA", "3PM" in Season 2000/2001:')
# first_3pp = avg_3p_data_ssn['3P%'][0]
# print("3P%", first_3pp)
# first_3pa = avg_3p_data_ssn['3PA'][0]
# print("3PA", first_3pa)
# first_3pm = avg_3p_data_ssn['3PM'][0]
# print("3PM", first_3pm)
# print('\n')

# # Average "3P%"", "3PA", "3PM" in Season 2020/2021

# print('Average "3P%"", "3PA", "3PM" in Season 2020/2021:')
# recent_3pp = avg_3p_data_ssn['3P%'][-1]
# print("3P%", recent_3pp)
# recent_3pa = avg_3p_data_ssn['3PA'][-1]
# print("3PA", recent_3pa)
# recent_3pm = avg_3p_data_ssn['3PM'][-1]
# print("3PM", recent_3pm)
# print('\n')

# df.loc[:, 'WINP'] = df.loc[:, 'WIN%']
# df.loc[:, 'TPP'] = df.loc[:, '3P%']
# df.loc[:, 'FGP'] = df.loc[:, 'FG%']
# df.loc[:, 'FTP'] = df.loc[:, 'FT%']
