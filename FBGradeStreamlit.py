import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import streamlit as st

def FBFunction(dataframe):
    number_columns = ['mins played', 'Yellow Card', 'Red Card', 'Goal', 'Assist', 'Dribble',
           'Goal Against', 'Stand. Tackle', 'Unsucc Stand. Tackle', 'Tackle',
           'Progr Rec', 'Unprogr Rec', 'Progr Inter', 'Unprogr Inter', 'Blocked Shot',
           'Blocked Cross', 'Att 1v1', 'Efforts on Goal',
           'Shot on Target', 'Att Shot Blockd', 'Cross', 'Unsucc Cross',
           'Long', 'Unsucc Long', 'Forward', 'Unsucc Forward', 'Line Break', 'Pass into Oppo Box',
           'Loss of Poss', 'Success', 'Unsuccess', 'Foul Won', 'Clear', 'Unsucc Tackle',
           'Foul Conceded', 'Progr Regain ', 'Stand. Tackle Success ', 'Pass Completion ', 'Progr Pass Completion ', 
           'PK Scored', 'PK Missed']


    details = dataframe.loc[:, ['Player Full Name', 'Team Name', 'Match Date', 'Position Tag', 'Starts']]
    #details = selected.loc[:, ['Player Full Name', 'Team Name', 'As At Date', 'Position Tag']]
    details.reset_index(drop=True, inplace=True)

    selected = dataframe.loc[:, ~dataframe.columns.duplicated()]
    selected_p90 = selected.loc[:, number_columns].astype(float)

    per90 = ['Yellow Card', 'Red Card', 'Goal', 'Assist', 'Dribble',
           'Stand. Tackle', 'Unsucc Stand. Tackle', 'Tackle', 'Clear', 'Unsucc Tackle',
           'Progr Rec', 'Unprogr Rec', 'Forward', 'Unsucc Forward', 'PK Missed', 'PK Scored']

    selected_p90['minutes per 90'] = selected_p90['mins played']/90

    for column in per90:
        if column not in ['Goal', 'Assist', 'Shot on Target', 'Yellow Card', 'Red Card', 'PK Missed', 'PK Scored']:
            selected_p90[column] = selected_p90[column] / selected_p90['minutes per 90']

    selected_p90 = selected_p90.drop(columns=['minutes per 90'])
    selected_p90.reset_index(drop=True, inplace=True)
    selected_p90.fillna(0, inplace=True)

    passing = selected_p90[['Pass Completion ']]
    passing.fillna(0, inplace=True)

    defending = selected_p90[['Progr Regain ']]
    defending.fillna(0, inplace=True)

    total_def_actions_columns = ['Tackle', 'Clear', 'Progr Inter', 'Unprogr Inter', 'Progr Rec', 
                             'Unprogr Rec', 'Stand. Tackle', 'Unsucc Stand. Tackle', 
                             'Unsucc Tackle']
    selected_p90['Total Def Actions'] = selected_p90[total_def_actions_columns].sum(axis=1)
    total_actions = selected_p90['Total Def Actions']
    total_actions.fillna(0, inplace=True)

    adjustments = selected_p90[['Yellow Card', 'Red Card', 'Goal', 'Assist', 'Pass into Oppo Box', 
                                'PK Missed', 'PK Scored']]

    for index, row in adjustments.iterrows():
        if adjustments['PK Scored'][index] == 1:
            adjustments['Goal'][index] = adjustments['Goal'][index] - 1
        elif adjustments['PK Scored'][index] == 2:
            adjustments['Goal'][index] = adjustments['Goal'][index] - 2
    adjustments.fillna(0, inplace=True)
    adjustments['Yellow Card'] = adjustments['Yellow Card'] * -.5
    adjustments['Red Card'] = adjustments['Red Card'] * -2
    adjustments['Goal'] = adjustments['Goal'] * 1
    adjustments['Assist'] = adjustments['Assist'] * 0.5
    adjustments['Pass into Oppo Box'] = adjustments['Pass into Oppo Box'] * 0.1
    adjustments['PK Missed'] = adjustments['PK Missed'] * -1
    adjustments['PK Scored'] = adjustments['PK Scored'] * 0.7

    def clip_percentile(value):
        return max(min(value, 100), 50)

    def calculate_percentile(value):
        return norm.cdf(value) * 100

    # Function to calculate z-score for each element in a column
    def calculate_zscore(column, mean, std):
        return (column - mean) / std

    player_location = []
    for index, row in details.iterrows():
        if 'RB' in row['Position Tag'] or 'LB' in row['Position Tag'] or 'RWB' in row['Position Tag'] or 'LWB' in row['Position Tag'] or 'WingB' in row['Position Tag']:
            player_location.append(index)
        if 'FB' in row['Position Tag']:
            player_location.append(index)

    readding = []
    selected_p90 = pd.concat([details, selected_p90], axis=1)
    final = pd.DataFrame()

    for i in player_location:
        more_data = selected_p90.iloc[i]
        player_name = more_data['Player Full Name']
        team_name = more_data['Team Name']
        date = more_data['Match Date']
        
        if team_name in ['Boston Bolts U13', 'Boston Bolts U14']:
            fb_df = pd.read_csv("Thresholds/FullBackThresholds1314.csv")
        elif team_name in ['Boston Bolts U15', 'Boston Bolts U16']:
            fb_df = pd.read_csv("Thresholds/FullBackThresholds1516.csv")
        elif team_name in ['Boston Bolts U17', 'Boston Bolts U19']:
            fb_df = pd.read_csv("Thresholds/FullBackThresholds1719.csv")




        mean_values = fb_df.iloc[0, 2]
        std_values = fb_df.iloc[1, 2]
        # Calculate the z-score for each data point
        z_scores_df = passing.transform(lambda col: calculate_zscore(col, mean_values, std_values))
        passing_percentile = z_scores_df.map(calculate_percentile)
        passing_percentile = passing_percentile.map(clip_percentile)
        passing_percentile = passing_percentile + 15
        player_passing = passing_percentile.iloc[player_location]
        weights = np.array([.1])
        passing_score = (
            player_passing['Pass Completion '] * weights[0]
            )   

        mean_values = fb_df.iloc[0, 1]
        std_values = fb_df.iloc[1, 1]
        # Calculate the z-score for each data point
        z_scores_df = defending.transform(lambda col: calculate_zscore(col, mean_values, std_values))
        defending_percentile = z_scores_df.map(calculate_percentile)
        defending_percentile = defending_percentile.map(clip_percentile)
        defending_percentile = defending_percentile + 15
        player_defending = defending_percentile.iloc[player_location]
        weights = np.array([.1])
        defending_score = (
            player_defending['Progr Regain '] * weights[0]
            )
        
        mean_values = fb_df.iloc[0, 0]
        std_values = fb_df.iloc[1, 0]
        # Calculate the z-score for each data point
        z_scores_df = total_actions.transform(lambda col: calculate_zscore(col, mean_values, std_values))
        total_actions_percentile = z_scores_df.map(calculate_percentile)
        total_actions_percentile = total_actions_percentile.map(clip_percentile)
        total_actions_percentile = total_actions_percentile + 15
        player_total_actions = total_actions_percentile.iloc[player_location].reset_index()
        weights = np.array([.1])
        total_actions_score = (
            player_total_actions['Total Def Actions'] * weights[0]
            )

        add = adjustments.iloc[i, :].sum()
        readding.append(add)

        final_grade = (defending_score * .2) + (passing_score * .2) + (total_actions_score * .2)
        final['Passing'] = passing_score
        final['Defending'] = defending_score
        final['Total Def Actions'] = total_actions_score
        final['Final Grade'] = final_grade
        final['Team Name'] = team_name
        final['Date'] = date


    player_name = []
    player_minutes = []
    player_position = []
    player_starts = []
    for i in player_location:
         player_name.append(selected_p90['Player Full Name'][i])
         player_minutes.append(selected_p90['mins played'][i])
         player_position.append(selected_p90['Position Tag'][i])
         player_starts.append(selected_p90['Starts'][i])
    final['Minutes'] = player_minutes
    final['Player Name'] = player_name
    final['Position'] = player_position
    final['Started'] = player_starts
    final['Adjustments'] = readding
    
    return final
