import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch


def goalkeeperHeatmap(df, gk_pname):
    df = df.loc[df['Player Full Name'] == gk_pname]

    df = df.dropna(subset=['x', 'y']).reset_index(drop=True)
    df[['x', 'y', 'ex', 'ey']] = df[['x', 'y', 'ex', 'ey']].astype(float)

    df = df[['Player Full Name', 'Team', 'Match Date', 'Opposition', 'Action', 'Period', 'Time',
            'x', 'y', 'Dir']]

    for index,row in df.iterrows():
        
        if row['Dir'] == 'RL':
            df.at[index, 'x'] = 120 - df.at[index, 'x']
            df.at[index, 'y'] = 80 - df.at[index, 'y']
        elif row['x'] < 0:
            df.at[index, 'x'] = 0
            
    dont_want = ['Line Break', 'Kick Off', 'FK']

    df = df.loc[~df['Action'].isin(dont_want)]

    # Remove duplicates, keeping the first occurrence
    df = df.drop_duplicates(subset=['x', 'y', 'Time'], keep='first')

    df['Where'] = ''

    for index, row in df.iterrows():
        if row['x'] <= 40:
            df.at[index, 'Where'] = 'First Third'
        elif row['x'] <= 80:
            df.at[index, 'Where'] = 'Second Third'
        else:
            df.at[index, 'Where'] = 'Final Third'
            
    first_third = round(df.loc[df['Where'] == 'First Third'].shape[0]/len(df) * 100, 0)
    second_third = round(df.loc[df['Where'] == 'Second Third'].shape[0]/len(df) * 100, 0)
    final_third = round(df.loc[df['Where'] == 'Final Third'].shape[0]/len(df) * 100, 0)

    fig, ax = plt.subplots(figsize=(13.5, 8))
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#FFFFFF", line_color="#000000")
    pitch.draw(ax=ax)
    plt.gca().invert_yaxis()

    sns.kdeplot(data=df, x='x', y='y', cmap='Blues', shade=True, ax=ax, bw_adjust=0.7, alpha=0.5)

    ax.annotate(f"First Third Actions\n{first_third}%", xy=(20, 40),
                fontsize=17, color='black', ha='center', va='center', fontfamily='Comic Sans MS', 
                fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))
    ax.annotate(f"Second Third Actions\n{second_third}%", xy=(60, 40),
                fontsize=17, color='black', ha='center', va='center', fontfamily='Comic Sans MS',
                fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))
    ax.annotate(f"Final Third Actions\n{final_third}%", xy=(100, 40),
                fontsize=17, color='black', ha='center', va='center', fontfamily='Comic Sans MS',
                fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))
    
    return fig