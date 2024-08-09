import pandas as pd
import os
import glob
import numpy as np
import streamlit as st

def averagesForEventData():
    # Directory paths for weekly reports
    weekly_report = 'WeeklyReport PSD/'
    
    # Initialize an empty list to store the selected dataframes
    selected_dfs = []
    
    def getFinalGrade(game_df):
        game_df.columns = game_df.iloc[3]
        game_df = game_df.iloc[4:]
        game_df = game_df.reset_index(drop=True)
    
        start_index = game_df.index[game_df["Period Name"] == "Round By Position Player"][0]
    
        # Find the index where "period name" is equal to "running by position player"
        end_index = game_df.index[game_df["Period Name"] == "Round By Team"][0]
    
    # Select the rows between the two indices
        selected_rows = game_df.iloc[start_index:end_index]
        selected = selected_rows.reset_index(drop=True)
        selected = selected.iloc[1:]    
        return selected
    
    # Function to process files in a folder
    def process_folder(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                game_df = pd.read_csv(file_path)
                selected_df = getFinalGrade(game_df)
                selected_dfs.append(selected_df)
    
    # readingn in all the weekly reports
    process_folder(weekly_report)
    
    # Concatenate all selected dataframes into one
    selected = pd.concat(selected_dfs, ignore_index=True)
    # selecting what we need
    selected = selected[['Player Full Name', 'Team Name', 'Opposition', 'Match Date', 'Position Tag', 'mins played', 'Success', 'Unsuccess']]
    selected[['mins played', 'Success', 'Unsuccess']] = selected[['mins played', 'Success', 'Unsuccess']].astype(float)
    
    grouped = selected.groupby(['Player Full Name', 'Position Tag'])['mins played'].sum().reset_index()
    
    # Find the position with the most minutes played for each player
    idx = grouped.groupby('Player Full Name')['mins played'].idxmax()
    prime_pos = grouped.loc[idx].reset_index(drop=True)
    prime_pos.rename(columns={'Position Tag': 'Primary Position'}, inplace=True)
    
    # getting the primary position
    selected = pd.merge(selected, prime_pos[['Player Full Name', 'Primary Position']], on='Player Full Name', how='inner')
    del selected['Position Tag']
    selected['Match Date'] = pd.to_datetime(selected['Match Date']).dt.strftime('%m/%d/%Y')
    # formatting to fit with the rest of the data
    selected.loc[selected['Opposition'] == 'St Louis', 'Match Date'] = '12/08/2023'
    selected = selected.groupby(['Player Full Name', 'Opposition', 'Match Date', 'Team Name', 'Primary Position'])[['mins played', 'Success', 'Unsuccess']].sum()
    selected.reset_index(inplace=True)
    
    
    # TEMPORARY PLACEHOLDER
    # Path to the folder containing CSV files
    folder_path = 'Actions PSD'
    
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    # List to hold individual DataFrames
    actions = pd.DataFrame()
    
    # Loop through the CSV files and read them into DataFrames
    for file in csv_files:
        df = pd.read_csv(file)
        columns = df.loc[4]
        df.columns = columns
        df = df.loc[5:].reset_index(drop=True)
        df = df[['Player Full Name', 'Match Date', 'Team', 'Opposition',
                         'Action', 'Period', 'Time', 'x', 'y', 'ex',  'ey', 'Dir']]
        if actions.empty:
            actions = pd.DataFrame(columns=columns)
    
        # Concatenate the current DataFrame with the actions DataFrame
        actions = pd.concat([actions, df], ignore_index=True)

    actions.reset_index(drop=True, inplace=True)
    
    # this function formats the data in a way that is consistent
    def gettingCorrectData(events):
        
        events = events[['Player Full Name', 'Match Date', 'Team', 'Opposition',
                         'Action', 'Period', 'Time', 'x', 'y', 'ex',  'ey', 'Dir']]
        locations = ['x', 'y', 'ex', 'ey']
        events[locations] = events[locations].astype(float)
        
        for index, row in events.iterrows():
            if row['Dir'] == 'RL':
                events.at[index, 'x'] = 120 - events.at[index, 'x']
                events.at[index, 'y'] = 80 - events.at[index, 'y']
                events.at[index, 'ex'] = 120 - events.at[index, 'ex']
                events.at[index, 'ey'] = 80 - events.at[index, 'ey']
                
        del events['Dir']
        return events
    
    actions = gettingCorrectData(actions)
    actions['Match Date'] = pd.to_datetime(actions['Match Date']).dt.strftime('%m/%d/%Y')
    
    # this finds the xT from passing and xT from dribbling separately 
    # we currently don't have xT from dribbling (need to talk with PSD)
    def finding_xT(overall_events, pass_or_dribble):
        overall_events.dropna(subset=['x'], inplace=True)
    
        overall_events = overall_events.loc[overall_events['Action'].isin(pass_or_dribble)]

        # filtering out rows that don't qualify for xT
        def filter_rows(row):
            if row['x'] == row['ex'] or row['y'] == row['ey']:
                return False  # Delete the row
            return True  # Keep the row
        
    
        # Apply the filter function to each row
        overall_events = overall_events[overall_events.apply(filter_rows, axis=1)].reset_index(drop=True)

        # these are the xT values for the grid
        xT = np.array([[0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267,
                0.01248344, 0.01473596, 0.0174506 , 0.02122129, 0.02756312,
                0.03485072, 0.0379259 ],
            [0.00750072, 0.00878589, 0.00942382, 0.0105949 , 0.01214719,
                0.0138454 , 0.01611813, 0.01870347, 0.02401521, 0.02953272,
                0.04066992, 0.04647721],
            [0.0088799 , 0.00977745, 0.01001304, 0.01110462, 0.01269174,
                0.01429128, 0.01685596, 0.01935132, 0.0241224 , 0.02855202,
                0.05491138, 0.06442595],
            [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646,
                0.01484598, 0.01689528, 0.0199707 , 0.02385149, 0.03511326,
                0.10805102, 0.25745362],
            [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646,
                0.01484598, 0.01689528, 0.0199707 , 0.02385149, 0.03511326,
                0.10805102, 0.25745362],
            [0.0088799 , 0.00977745, 0.01001304, 0.01110462, 0.01269174,
                0.01429128, 0.01685596, 0.01935132, 0.0241224 , 0.02855202,
                0.05491138, 0.06442595],
            [0.00750072, 0.00878589, 0.00942382, 0.0105949 , 0.01214719,
                0.0138454 , 0.01611813, 0.01870347, 0.02401521, 0.02953272,
                0.04066992, 0.04647721],
            [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267,
                0.01248344, 0.01473596, 0.0174506 , 0.02122129, 0.02756312,
                0.03485072, 0.0379259 ]])
        xT_rows, xT_cols = xT.shape
        # setting it up so that x, y, ex, ey have an equal amount of random points in them
        np.random.seed(42)
        x = np.random.uniform(0, 120, 240)
        y = np.random.uniform(0, 80, 240)
        ex = np.random.uniform(0, 120, 240)
        ey = np.random.uniform(0, 80, 240)

        # binning based on the data
        add_data = pd.DataFrame({'x': x, 
                            'y': y,
                            'ex': ex,
                            'ey': ey})
        x_bins = pd.cut(add_data['x'], xT_cols, retbins=True)[1]
        y_bins = pd.cut(add_data['y'], xT_rows, retbins=True)[1]
        ex_bins = pd.cut(add_data['ex'], xT_cols, retbins=True)[1]
        ey_bins = pd.cut(add_data['ey'], xT_rows, retbins=True)[1]
    
        overall_events['x1_bin'] = pd.cut(overall_events['x'], bins=x_bins, labels=False, include_lowest=True)
        overall_events['y1_bin'] = pd.cut(overall_events['y'], bins=y_bins, labels=False, include_lowest=True)
        overall_events['x2_bin'] = pd.cut(overall_events['ex'], bins=ex_bins, labels=False, include_lowest=True)
        overall_events['y2_bin'] = pd.cut(overall_events['ey'], bins=ey_bins, labels=False, include_lowest=True)
    
        overall_events.dropna(subset=['x1_bin', 'y1_bin', 'x2_bin', 'y2_bin'], inplace=True)
        overall_events.reset_index(drop=True)
        overall_events['y1_bin'] = overall_events['y1_bin'].astype(int)
        overall_events['y2_bin'] = overall_events['y2_bin'].astype(int)
        overall_events['x1_bin'] = overall_events['x1_bin'].astype(int)
        overall_events['x2_bin'] = overall_events['x2_bin'].astype(int)

        # getting the start and end values
        overall_events['start_zone_value'] = overall_events[['x1_bin', 'y1_bin']].apply(lambda x: xT[x[1], x[0]], axis=1)
        overall_events['end_zone_value'] = overall_events[['x2_bin', 'y2_bin']].apply(lambda x: xT[x[1], x[0]], axis=1)
    
        # this is xT
        overall_events['xT'] = overall_events['end_zone_value'] - overall_events['start_zone_value']
    
        # getting the xT for a unique player during a game
        player_sum = overall_events.groupby(['Player Full Name', 'Team', 'Opposition', 'Match Date'])['xT'].sum()
        player_sum = player_sum.to_frame()
        player_sum.reset_index(inplace=True)
        if 'Forward' in pass_or_dribble:
            player_sum.rename(columns={'xT': 'xT Passing'})
        else:
            player_sum.rename(columns={'xT': 'xT Dribbling'})
        return player_sum
    
    dribbling = ['Dribble']
    passing = ['Forward', 'Side Back', 'Cross', 'Long', 'Ground GK', 'Throw']
    # This isn't working
    #xT_dribbling = finding_xT(overall_events=actions, pass_or_dribble=dribbling)
    xT_passing = finding_xT(overall_events=actions, pass_or_dribble=passing)
    
    def finalThird(events_third):
        passes = ['Forward', 'Side Back', 'Unsucc Forward', 'Unsucc Side Back', 
                  'Unsucc Cross', 'Cross', 'Long', 'Unsucc Long']
        dribbles = ['Dribble', 'Loss of Poss']
        touches = ['Forward', 'Side Back', 'Cross', 'Long', 'Dribble', 
                  'Progr Rec', 'Progr Inter', 'Foul Won', 'Def Aerial', 'Att Aerial', 
                  'Shot on Target', 'Shot off Target', 'Stand Tackle', 'Att Shot Blockd',
                  'Tackle', 'Short Corner', 'Corner Kick', 'Unsucc Corner Kick', 
                  'Header on Target', 'Header off Target']

        # final third touches
        touches_df = events_third.loc[events_third['Action'].isin(touches)]
        
        # final third dribbles and passes
        dribblePass = events_third.loc[events_third['Action'].isin(passes+dribbles)]
        dribblePass.dropna(inplace=True)

        # making sure they're in the final third
        dribblePass = dribblePass.loc[dribblePass['ex'] >= 80]
        touches_df = touches_df.loc[touches_df['ex'] >= 80]

        # getting unique touches
        touches_df = touches_df.groupby(['Player Full Name', 
                                             'Team', 'Opposition', 'Match Date']).size().reset_index(name='Final Third Touches')
    
    
        def categorize_action(action, desired_list):
            if action in desired_list:
                return 1
            else:
                return 0
        
        # completed passes
        complete_pass = ['Forward', 'Side Back', 'Cross', 'Long']
        # uncompleted passes
        incomplete_pass = ['Unsucc Forward', 'Unsucc Side Back', 
                  'Unsucc Cross', 'Unsucc Long']
        complete_drib = ['Dribble']
        incomplete_drib = ['Loss of Poss']
        
        # Apply categorization function to create 'Completed Pass' column
        dribblePass['Complete Pass'] = dribblePass['Action'].apply(categorize_action, desired_list=complete_pass)
        dribblePass['Incomplete Pass'] = dribblePass['Action'].apply(categorize_action, desired_list=incomplete_pass)
        dribblePass['Complete Dribble'] = dribblePass['Action'].apply(categorize_action, desired_list=complete_drib)
        dribblePass['Incomplete Dribble'] = dribblePass['Action'].apply(categorize_action, desired_list=incomplete_drib) 
        dribblePass['Final Third Passes'] = dribblePass['Complete Pass'] + dribblePass['Incomplete Pass']
    
        end_dribblePass = dribblePass[['Player Full Name', 'Team', 'Opposition', 'Match Date',
                                       'Complete Pass', 'Incomplete Pass', 'Complete Dribble', 
                                       'Incomplete Dribble', 'Final Third Passes']]
        end_result = end_dribblePass.groupby(['Player Full Name', 'Team', 'Opposition', 'Match Date']).sum()
        end_result.reset_index(inplace=True)
    
        end_result['Pass %'] = (end_result['Complete Pass']/(end_result['Complete Pass'] + end_result['Incomplete Pass'])) * 100
        end_result['Dribble %'] = (end_result['Complete Dribble']/(end_result['Complete Dribble'] + end_result['Incomplete Dribble']))*100

        # doing an outer join because there can be touches without end results
        end_result = pd.merge(touches_df, end_result, on=['Player Full Name', 'Team', 'Opposition', 'Match Date'], how='outer')
        return(end_result)
    dribblePass_df = finalThird(actions)
    
    # doing the same outer join here
    pd_result = pd.merge(dribblePass_df, xT_passing, on=['Player Full Name', 'Team', 'Opposition', 'Match Date'], how='outer')
    
    selected.rename(columns={'Team Name': 'Team'}, inplace=True)
    final_result = pd.merge(selected, pd_result, on=['Player Full Name', 'Team', 'Opposition', 'Match Date'], how='outer')
    final_result['Total Passes'] = final_result['Success'] + final_result['Unsuccess']
    
    # converting everything to p90 that needs ot be
    for col in ['Final Third Passes', 'Final Third Touches', 'xT', 'Total Passes']:
        final_result[f'{col}'] = final_result[col] / final_result['mins played'] * 90
        
    # xT per Pass for GKs
    final_result['xT per Pass'] = final_result['xT']/final_result['Total Passes']
    
    # selecting everything that we need
    final_result = final_result[['Player Full Name', 'Team', 'Opposition', 'Match Date', 'mins played', 'Primary Position',
                                 'xT per Pass', 'Final Third Passes', 'Final Third Touches', 'xT']]
    
    final_result.loc[final_result['Opposition'] == 'St Louis', 'Match Date'] = '12/09/2023'

    return final_result

