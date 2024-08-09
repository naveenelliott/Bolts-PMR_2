import pandas as pd
import streamlit as st
from GettingPSDGradeData import getting_PSD_grade_data
import matplotlib.image as mpimg
import base64
import glob
import os
from xGModel import xGModel
from mplsoccer import VerticalPitch, Pitch
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from GoalkeeperHeatmap import goalkeeperHeatmap
from GKGradeStreamlit import GKMoreDetailedFunction
from GettingEventDataGrades import GKEventFunction
from EventDataGradeStreamlit import averagesForEventData
import scipy.stats as stats
from PlottingGKReport import plottingStatistics, gettingGameGrade, gkInvolvements
from plotly_football_pitch import make_pitch_figure, PitchDimensions
import plotly_football_pitch as pfp
import plotly.graph_objs as go

#st.set_page_config(layout='wide')

# adding in data from PMRApp
combined_df = st.session_state["combined_df"]
end_combined_df = combined_df.copy()
overall_df = st.session_state['overall_df'] 
selected_team = st.session_state["selected_team"]
selected_opp = st.session_state["selected_opp"]
selected_date = st.session_state["selected_date"]

gk_data = st.session_state['gk_df']

gk_name = st.session_state["selected_gk"]
gk_data = gk_data.loc[gk_data['Player Full Name'] == gk_name]
gk_data.drop(columns=['In Possession', 'Out Possession'], inplace=True)

in_n_out_df = pd.read_csv('pages/InAndOutOfPossessionGoalsGK.csv')

gk_info = pd.merge(gk_data, in_n_out_df, on=['Player Full Name', 'Date', 'Opposition', 'Team Name'], how='inner')
gk_info.reset_index(drop=True, inplace=True)

if not pd.isna(gk_info['Vasily Notes']).any() and not gk_info.empty:
    # Extract relevant columns into variables
    in_possession_goals = gk_info['In Possession'].iloc[0]
    out_possession_goals = gk_info['Out Possession'].iloc[0]
    coach_notes = gk_info['Vasily Notes'].iloc[0]
    url = gk_info['Veo Hyperlink'].iloc[0]

    st.title(f"{gk_name} - Goalkeeper Report ({selected_team} vs {selected_opp})")

    gk_data = getting_PSD_grade_data()
    gk_data.loc[gk_data['Player Full Name'] == 'Casey Powers', 'Position Tag'] = 'GK'
    gk_data = gk_data.loc[gk_data['Position Tag'] == 'GK']
    non_number_columns = ['Player Full Name', 'Team Name', 'Position Tag', 'Match Date', 'Opposition']
    for col in gk_data.columns:
        if col not in non_number_columns:
            gk_data[col] = gk_data[col].astype(float)
    gk_data['Match Date'] = pd.to_datetime(gk_data['Match Date'])
    gk_data['Match Date'] = gk_data['Match Date'].dt.strftime('%m/%d/%Y')

    # fromatting for the selected game
    all_games_gk = gk_data.copy()
    gk_data = gk_data.loc[(gk_data['Team Name'] == selected_team) & (gk_data['Opposition'] == selected_opp) & (gk_data['Match Date'] == selected_date)].reset_index(drop=True)
    overall_gk_data = gk_data.copy()
    specific_player = gk_data.loc[gk_data['Player Full Name'] == gk_name]

    in_poss_involve, out_poss_involve = gkInvolvements(specific_player)


    

    yellow_cards = gk_data['Yellow Card'][0]
    red_cards = gk_data['Red Card'][0]
    mins_played = gk_data['mins played'][0]

    col1, col2 = st.columns(2)

    folder_path = 'xG Input Files'

    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # List to hold individual DataFrames
    df_list = []

    # Loop through the CSV files and read them into DataFrames
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    fc_python = pd.concat(df_list, ignore_index=True)

    # Making sure everything is aligned on one side
    def flip_coordinates(x, y):
        # Flip x and y coordinates horizontally
        flipped_x = field_dim - x
        flipped_y = field_dim - y  # y remains unchanged in this transformation
        
        return flipped_x, flipped_y

    field_dim = 100
    # Iterating through coordinates and making them on one side
    flipped_points = []
    for index, row in fc_python.iterrows():
        if row['X'] < 50:
            flipped_x, flipped_y = flip_coordinates(row['X'], row['Y'])
            fc_python.at[index, 'X'] = flipped_x
            fc_python.at[index, 'Y'] = flipped_y


    # Path to the folder containing CSV files
    folder_path = 'Actions PSD'

    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # List to hold individual DataFrames
    df_list = []

    # Loop through the CSV files and read them into DataFrames
    for file in csv_files:
        df = pd.read_csv(file)
        df.columns = df.loc[4]
        df = df.loc[5:].reset_index(drop=True)
        df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    actions = pd.concat(df_list, ignore_index=True)
    actions['Match Date'] = pd.to_datetime(actions['Match Date']).dt.strftime('%m/%d/%Y')
    actions.loc[actions['Opposition'] == 'St Louis', 'Match Date'] = '12/09/2023'

    # creating copies to work on
    full_actions = actions.copy()
    entire_actions = actions.copy()

    full_actions = full_actions.loc[(full_actions['Team'] == selected_team) & (full_actions['Opposition'] == selected_opp) & (full_actions['Match Date'] == selected_date)].reset_index(drop=True)
    full_actions_copy = full_actions.copy()
    select_event_df = full_actions.copy()

    # these are the ideal columns
    cols = ['Player Full Name', 'Team', 'Match Date', 'Opposition', 'Action', 'Time', 'Video Link']
    xg_actions = actions[cols]

    # these are the shots we want
    wanted_actions = ['Att Shot Blockd', 'Blocked Shot', 'Goal', 'Goal Against', 'Header on Target', 
                    'Header off Target', 'Opp Effort on Goal', 'Save Held', 'Save Parried', 'Shot off Target', 
                    'Shot on Target']
    xg_actions = xg_actions.loc[xg_actions['Action'].isin(wanted_actions)].reset_index(drop=True)
    # renaming for the join
    xg_actions.rename(columns={'Team': 'Bolts Team'}, inplace=True)

    # if the opponent shots are 0, we factor in blocked shots
    # if they aren't 0, then we don't factor in blocked shots
    opp_shots = gk_data['Opp Effort on Goal'].sum()
    temp_shots = opp_shots.copy()
    if temp_shots != 0:
        xg_actions = xg_actions.loc[xg_actions['Action'] != 'Blocked Shot'].reset_index(drop=True)

    # this is handeling duplicated PlayerStatData shots 
    temp_df = pd.DataFrame(columns=xg_actions.columns)
    prime_actions = ['Opp Effort on Goal', 'Shot on Target']
    remove_indexes = []
    for index in range(len(xg_actions) - 1):
        if xg_actions.loc[index, 'Time'] == xg_actions.loc[index+1, 'Time']:
            temp_df = pd.concat([temp_df, xg_actions.loc[[index]], xg_actions.loc[[index + 1]]], ignore_index=False)
            bye1 = temp_df.loc[temp_df['Action'].isin(prime_actions)]
            # these are the indexes we want to remove
            remove_indexes.extend(bye1.index)
            
        temp_df = pd.DataFrame(columns=xg_actions.columns)     

    # this is a copy with the removed duplicated PSD shots
    actions_new = xg_actions.copy()
    actions_new = actions_new.drop(remove_indexes).reset_index(drop=True) 

    fc_python['Match Date'] = pd.to_datetime(fc_python['Match Date']).dt.strftime('%m/%d/%Y')

    # combining into xG dataframe we want
    combined_xg = pd.merge(fc_python, actions_new, on=['Bolts Team', 'Match Date', 'Time'], how='inner')

    # running the model on our dataframe
    xg = xGModel(combined_xg)
    entire_xg = xg.copy()
    xg = xg.loc[(xg['Bolts Team'] == selected_team) & (xg['Opposition'] == selected_opp) & (xg['Match Date'] == selected_date)]
    xg = xg[['Team', 'X', 'Y', 'xG', 'Event', 'Time', 'Video Link']]



    if len(overall_gk_data) > 1:
        starting_gk = overall_gk_data.loc[overall_gk_data['Starts'] == 1].reset_index(drop=True)
        starting_gk_mins = starting_gk['mins played'][0]
        starting_gk_name = starting_gk['Player Full Name'][0]
        other_gk = overall_gk_data.loc[overall_gk_data['Player Full Name'] != starting_gk_name].reset_index(drop=True)
        other_gk_name = other_gk['Player Full Name'][0]

        starting_xg = pd.DataFrame(columns=xg.columns)
        other_xg = pd.DataFrame(columns=xg.columns)

        starting_xg_list = []
        other_xg_list = []

        for index, row in xg.iterrows():
            if row['Time'] <= starting_gk_mins:
                starting_xg_list.append(row)
            else:
                other_xg_list.append(row)

        starting_xg = pd.concat([starting_xg, pd.DataFrame(starting_xg_list)], ignore_index=True)
        starting_xg['Player Full Name'] = starting_gk_name
        other_xg = pd.concat([other_xg, pd.DataFrame(other_xg_list)], ignore_index=True)
        other_xg['Player Full Name'] = other_gk_name


        if gk_name in starting_xg['Player Full Name'].values:
            xg = starting_xg.copy()
        elif gk_name in other_xg['Player Full Name'].values:
            xg = other_xg.copy()



    custom_order = ['Shot', 'Blocked', 'SOT', 'Goal']
    xg['Event'] = pd.Categorical(xg['Event'], categories=custom_order, ordered=True)
    xg = xg.sort_values('Event')

    xg = xg.loc[~xg['Team'].str.contains('Boston Bolts')]
    xg_sum = xg['xG'].sum()
    ga = gk_data['Goal Against'][0].astype(float)

    dimensions = PitchDimensions(pitch_length_metres=100, pitch_width_metres=100)
    fig1 = pfp.make_pitch_figure(
        dimensions,
        figure_height_pixels=500,
        figure_width_pixels=500,
        orientation=pfp.PitchOrientation.VERTICAL
    )

    for index, row in xg.iterrows():
        y, x, xG, url = row['X'], row['Y'], row['xG'], row['Video Link']

        ateam = row['Team']
        if row['Event'] == 'Goal':
            fig1.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=(xG * 30) + 5,  # Adjusted for Plotly's scaling
                    color='red',
                    symbol='circle'
                ),
                name='Goal Against',
                showlegend=False,
                hovertext=f"xG: {xG:.2f}",  # Add hover text
                hoverinfo="text"  # Display hover text
            ))
            fig1.add_annotation(
                x=x,
                y=y+3.5,
                text=f'<a href="{url}" target="_blank" style="color:red;">Link</a>',
                showarrow=False,
                align="center"
            ) 
        elif row['Event'] == 'SOT':
            fig1.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=(xG * 30) + 5,
                    color='white',
                    line=dict(color='red', width=3),
                    symbol='circle'
                ),
                name='SOT Against',
                showlegend=False,
                hovertext=f"xG: {xG:.2f}",  # Add hover text
                hoverinfo="text"  # Display hover text
            ))
            fig1.add_annotation(
                x=x,
                y=y+3.5,
                text=f'<a href="{url}" target="_blank" style="color:red;">Link</a>',
                showarrow=False,
                align="center"
            ) 
        else:
            fig1.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=(xG * 30) + 5,
                    color='red',
                    symbol='circle-open'
                ),
                name='Shot Against',
                showlegend=False,
                hovertext=f"xG: {xG:.2f}",  # Add hover text
                hoverinfo="text"  # Display hover text
                ))
            fig1.add_annotation(
                x=x,
                y=y+3.5,
                text=f'<a href="{url}" target="_blank" style="color:red;">Link</a>',
                showarrow=False,
                align="center"
            ) 
        
    # Add custom legend entries
    fig1.add_trace(go.Scatter(
        x=[None],  # Dummy data
        y=[None],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            symbol='circle'
        ),
        name='Goal Against',
        visible='legendonly'
    ))

    fig1.add_trace(go.Scatter(
        x=[None],  # Dummy data
        y=[None],
        mode='markers',
        marker=dict(
            size=5,
            color='white',
            line=dict(color='red', width=3),
            symbol='circle'
        ),
        name='SOT Against',
        visible='legendonly'
    ))

    fig1.add_trace(go.Scatter(
        x=[None],  # Dummy data
        y=[None],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            symbol='circle-open'
        ),
        name='Shot Against',
        visible='legendonly',
    ))

    fig1.update_layout(
        legend=dict(
            font=dict(
                size=8  # Decrease font size for smaller legend text
            ),
            itemsizing='constant',  # Keep marker sizes constant in the legend
            traceorder='normal'  # Keep the order of traces as added
        )
    )



    player_pic = mpimg.imread('GK_Photos/temp.png')
    #yellow_card = mpimg.imread('pages/Yellow_Card.png')
    #red_card = mpimg.imread('pages/Red_Card.png')

    with col1:
        inner_columns = st.columns(2)

        with inner_columns[0]:
            st.image(player_pic, width=125)
            st.markdown(
            """
            <div style='display: block; text-align: left;'>
                <span style='font-family: Arial; font-size: 10pt; color: black;'>Player Name: {gk_name}</span>
            </div>
            """.format(gk_name=gk_name),
            unsafe_allow_html=True
        )
            st.markdown(
            """
            <div style='display: block; text-align: left;'>
                <span style='font-family: Arial; font-size: 10pt; color: black;'>Minutes Played: {vars}</span>
            </div>
            """.format(vars=mins_played),
            unsafe_allow_html=True
        )
            st.markdown(
            """
            <div style='display: block; text-align: left;'>
                <span style='font-family: Arial; font-size: 10pt; color: black;'>Yellow Cards: {vars}</span>
            </div>
            """.format(vars=yellow_cards),
            unsafe_allow_html=True
        )
            st.markdown(
            """
            <div style='display: block; text-align: left;'>
                <span style='font-family: Arial; font-size: 10pt; color: black;'>Red Cards: {vars}</span>
            </div>
            """.format(vars=red_cards),
            unsafe_allow_html=True
        )
        
        with inner_columns[1]:
            image_path = "pages/Veo.jpg"  # Replace with the path to your image

            def load_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            # Get the base64-encoded image
            image_base64 = load_image(image_path)

            # HTML and CSS for making the image a clickable link
            st.markdown(
            f"""
            <a href="{url}" target="_blank">
                <img src="data:image/jpeg;base64,{image_base64}" style="cursor: pointer; width: 75px;"/>  <!-- Adjust width as needed -->
            </a>
            """,
            unsafe_allow_html=True
            )

            st.markdown(
                """
                <div style='display: block; text-align: left;'>
                    <span style='font-family: Arial; font-size: 10pt; color: black;'>In Possession Involvements: {in_possession_involve}</span><br>
                    <span style='font-family: Arial; font-size: 10pt; color: black;'>Out of Possession Involvements: {out_possession_involve}</span>
                </div>
                """.format(in_possession_involve=in_poss_involve,
                           out_possession_involve=out_poss_involve),
                unsafe_allow_html=True
            )

    with col1:
        st.plotly_chart(fig1)

    with col2:
        fig = goalkeeperHeatmap(full_actions_copy, gk_name)
        st.pyplot(fig)


    gk_grade = GKMoreDetailedFunction(gk_data)
    gk_grade = gk_grade.loc[gk_grade['Player Name'] == gk_name].reset_index(drop=True)

    all_event_df = averagesForEventData()

    select_event_df = all_event_df.loc[(all_event_df['Team'] == selected_team) & (all_event_df['Opposition'] == selected_opp) & (all_event_df['Match Date'] == selected_date)]

    temp_event_df = all_event_df.loc[(all_event_df['Primary Position'] == 'GK')]
    temp_event_df = temp_event_df[['xT per Pass', 'Team']]
    select_temp_df = select_event_df.loc[select_event_df['Player Full Name'] == gk_name]
    select_temp_df = select_temp_df[['xT per Pass', 'Team']]
    xT_per_pass_grade = GKEventFunction(temp_event_df, select_temp_df)
    xT_per_pass_grade = xT_per_pass_grade.at[0, 'xT per Pass'] * 0.1


    ga_mean = -0.0794
    ga_std = 0.2558

    xga_grade = ga - xg_sum
    z_score = (xga_grade - ga_mean) / ga_std

    # Step 2: Convert Z-score to percentile
    ga_percentile = stats.norm.cdf(z_score) * 100
    ga_percentile = 100-ga_percentile

    ga_grade = ga_percentile * 0.1

    # This is until the claiming part of the grade is fixed
    if gk_grade['Defending Space'].isna().any():
        gk_grade.at[0, 'Attacking'] = (gk_grade.at[0, 'Attacking']*0.6) + (xT_per_pass_grade*0.4)
        gk_grade.at[0, 'Defending Goal'] = (gk_grade.at[0, 'Defending Goal']*0.25)+(ga_grade*.75)
        gk_grade.at[0, 'Final Grade'] = (gk_grade.at[0, 'Attacking']*0.3)+(gk_grade.at[0, 'Defending Goal']*0.5)+(gk_grade.at[0, 'Organization']*.2)
    else:
        gk_grade.at[0, 'Attacking'] = (gk_grade.at[0, 'Attacking']*0.6) + (xT_per_pass_grade*0.4)
        gk_grade.at[0, 'Defending Goal'] = (gk_grade.at[0, 'Defending Goal']*0.25)+(ga_grade*.75)
        gk_grade.at[0, 'Final Grade'] = (gk_grade.at[0, 'Attacking']*0.2375)+(gk_grade.at[0, 'Defending Goal']*0.4375)+(gk_grade.at[0, 'Organization']*.1375)+(gk_grade.at[0, 'Defending Space']*.1875)



    def process_game_data(df, overall_gk_data, gk_name):
        # Filter data for the specific game
        xg = df[['Team', 'X', 'Y', 'xG', 'Event', 'Time', 'Bolts Team']]

        if len(overall_gk_data) > 1:
            starting_gk = overall_gk_data.loc[overall_gk_data['Starts'] == 1].reset_index(drop=True)
            starting_gk_mins = starting_gk['mins played'][0]
            starting_gk_name = starting_gk['Player Full Name'][0]
            other_gk = overall_gk_data.loc[overall_gk_data['Player Full Name'] != starting_gk_name].reset_index(drop=True)
            other_gk_name = other_gk['Player Full Name'][0]

            starting_xg = pd.DataFrame(columns=xg.columns)
            other_xg = pd.DataFrame(columns=xg.columns)

            starting_xg_list = []
            other_xg_list = []

            for index, row in xg.iterrows():
                if row['Time'] <= starting_gk_mins:
                    starting_xg_list.append(row)
                else:
                    other_xg_list.append(row)

            starting_xg = pd.concat([starting_xg, pd.DataFrame(starting_xg_list)], ignore_index=True)
            starting_xg['Player Full Name'] = starting_gk_name
            other_xg = pd.concat([other_xg, pd.DataFrame(other_xg_list)], ignore_index=True)
            other_xg['Player Full Name'] = other_gk_name

            if gk_name in starting_xg['Player Full Name'].values:
                xg = starting_xg.copy()
            elif gk_name in other_xg['Player Full Name'].values:
                xg = other_xg.copy()
        else:
            xg['Player Full Name'] = gk_name


        xg.reset_index(drop=True, inplace=True)

        # Summing xG
        summed_xg = xg['xG'].sum()

        # Creating summary row
        summary = {
            'Player Full Name': xg['Player Full Name'].iloc[0],
            'Team': xg['Bolts Team'].iloc[0],
            'Match Date': df['Match Date'].iloc[0],
            'Opposition': df['Opposition'].iloc[0],
            'xG': summed_xg
        }

        return summary

    # Apply the function to each game
    processed_data = []


    entire_xg = entire_xg.loc[~entire_xg['Team'].str.contains('Boston Bolts')]
    for (team, opponent, match_date), group in entire_xg.groupby(['Bolts Team', 'Opposition', 'Match Date']):
        # Assuming overall_gk_data and gk_name are available for each group
        temp_game_gk = all_games_gk.loc[(all_games_gk['Team Name'] == team) & (all_games_gk['Opposition'] == opponent) & 
                                        (all_games_gk['Match Date'] == match_date)]
        processed_group = process_game_data(group, temp_game_gk, gk_name)
        processed_data.append(processed_group)

    # Combine all processed groups into one DataFrame
    summary_df = pd.DataFrame(processed_data)


    xT_every_game = all_event_df.loc[all_event_df['Player Full Name'] == gk_name]
    del xT_every_game['Final Third Passes'], xT_every_game['Final Third Touches'], xT_every_game['xT']

    end_combined_df = overall_df.loc[overall_df['Player Full Name'] == gk_name]
    unique_combinations = end_combined_df[['Team Name', 'Opposition', 'Date']].drop_duplicates()
    unique_combinations.rename(columns={'Date': 'Match Date'}, inplace=True)


    # Step 2: Filter all_games_gk by these combinations
    end_overall = all_games_gk.merge(unique_combinations, on=['Team Name', 'Opposition', 'Match Date'], how='inner')
    end_overall = end_overall.loc[end_overall['Player Full Name'] == gk_name]
    end_overall = end_overall[['Player Full Name', 'Team Name', 'Opposition', 'Match Date', 'Save Held', 'Save Parried', 'Goal Against', 'Progr Regain ', 
                            'Pass Completion ', 'Opp Effort on Goal']]
    end_overall['Save %'] = (end_overall['Save Held']+end_overall['Save Parried'])/(end_overall['Save Held']+end_overall['Save Parried']+end_overall['Goal Against'])*100
    game_grade_end = end_overall.copy()
    end_overall.drop(columns=['Save Held', 'Save Parried'], inplace=True)
    end_overall.rename(columns={'Team Name': 'Team'}, inplace=True)
    game_grade_end.rename(columns={'Team Name': 'Team'}, inplace=True)

    end_overall = pd.merge(end_overall, summary_df, on=['Player Full Name', 'Team', 'Opposition', 'Match Date'], how='inner')
    end_overall = pd.merge(end_overall, xT_every_game, on=['Player Full Name', 'Team', 'Opposition', 'Match Date'], how='inner')
    end_overall['GA-xGA'] = end_overall['Goal Against'] - end_overall['xG']
    del end_overall['Goal Against'], end_overall['xG']
    end_overall = end_overall[end_overall['Match Date'] <= selected_date]

    game_grade_end = pd.merge(game_grade_end, summary_df, on=['Player Full Name', 'Team', 'Opposition', 'Match Date'], how='inner')
    game_grade_end = pd.merge(game_grade_end, xT_every_game, on=['Player Full Name', 'Team', 'Opposition', 'Match Date'], how='inner')
    game_grade_end['GA-xGA'] = game_grade_end['Goal Against'] - game_grade_end['xG']


    final_game_grade = pd.DataFrame(columns=['Player Full Name', 'Match Date', 'Team', 'Opposition', 'Final Grade'])

    for _, row in unique_combinations.iterrows():
        match_date = row['Match Date']
        team_name = row['Team Name']
        opposition = row['Opposition']

        select_event_df = all_event_df.loc[(all_event_df['Team'] == team_name) & 
                                        (all_event_df['Opposition'] == opposition) & (all_event_df['Match Date'] == match_date)]
        select_temp_df = select_event_df.loc[select_event_df['Player Full Name'] == gk_name]
        select_temp_df = select_temp_df[['xT per Pass', 'Team']]

        
        # Filter the DataFrame for the current combination
        filtered_game_grade = game_grade_end[(game_grade_end['Match Date'] == match_date) & 
                        (game_grade_end['Team'] == team_name) & 
                        (game_grade_end['Opposition'] == opposition)]
        filtered_game_grade = filtered_game_grade.loc[filtered_game_grade['Player Full Name'] == gk_name]
        
        # Assuming gettingGameGrade and GKEventFunction are functions that operate on the filtered_df
        xT_temp_grade = GKEventFunction(temp_event_df, select_temp_df)
        xT_temp_grade = xT_temp_grade.at[0, 'xT per Pass']
        game_grade = gettingGameGrade(filtered_game_grade, xT_temp_grade)

        final_game_grade = pd.concat([final_game_grade, game_grade], ignore_index=True)
        

    end_overall = end_overall.sort_values('Match Date').reset_index(drop=True)
    final_game_grade = final_game_grade.sort_values('Match Date').reset_index(drop=True)
    final_game_grade = final_game_grade[final_game_grade['Match Date'] <= selected_date]

    

    
    fig = plottingStatistics(final_game_grade, 'Final Grade', date_wanted=selected_date)
    st.plotly_chart(fig)

    with col1:
        fig = plottingStatistics(end_overall, 'GA-xGA', date_wanted=selected_date)
        st.plotly_chart(fig)
        fig2 = plottingStatistics(end_overall, 'xT per Pass', date_wanted=selected_date)
        st.plotly_chart(fig2)
    with col2:
        fig = plottingStatistics(end_overall, 'Save %', date_wanted=selected_date)
        st.plotly_chart(fig)
        fig2 = plottingStatistics(end_overall, 'Pass Completion ', date_wanted=selected_date)
        st.plotly_chart(fig2)

    st.markdown(
            """
            <div style='display: block; text-align: left;'>
                <span style='font-family: Arial; font-size: 13pt; color: black;'><strong>In Possession Goals:</strong> {in_possession_goals}</span><br><br>
                <span style='font-family: Arial; font-size: 13pt; color: black;'><strong>Out of Possession Goals:</strong> {out_possession_goals}</span><br><br>
                <span style='font-family: Arial; font-size: 13pt; color: black;'><strong>Coach Notes:</strong> {coach_notes}</span>
            </div>
            """.format(in_possession_goals=in_possession_goals, 
                    out_possession_goals=out_possession_goals, 
                    coach_notes=coach_notes),
            unsafe_allow_html=True
        )

else:
    st.title("Goalkeeper Report is NOT AVAILABLE. Please fill out the coach requirements")
