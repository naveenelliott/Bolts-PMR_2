import streamlit as st
import pandas as pd
from GettingFullActions import UpdatingActions
from GettingPSDLineupData import getting_PSD_lineup_data

# Setting the title of the PMR App in web browser
st.set_page_config(page_title='Bolts Post-Match Review App', page_icon='pages/Boston_Bolts.png')

st.sidebar.success('Select a page above.')

# This updates actions
combined_actions = UpdatingActions()

# These are the allowable teams that we have event data for
bolts_allowed = pd.Series(combined_actions['Team'].unique())
opp_allowed = pd.Series(combined_actions['Opposition'].unique())

# Getting and processing lineup data
combined_df = getting_PSD_lineup_data()
combined_df['Starts'] = combined_df['Starts'].astype(float)
combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.strftime('%m/%d/%Y')
combined_df = combined_df.loc[combined_df['Team Name'].isin(bolts_allowed) & combined_df['Opposition'].isin(opp_allowed)].reset_index(drop=True)

# Handling special cases for player positions
combined_df.loc[combined_df['Player Full Name'] == 'Casey Powers', 'Position Tag'] = 'GK'

# Goalkeeper-specific DataFrame
gk_dataframe = combined_df.loc[combined_df['Position Tag'] == 'GK'].reset_index(drop=True).drop_duplicates().reset_index(drop=True)
st.session_state['complete_gk_df'] = gk_dataframe.copy()

# Creating a transferrable copy of the combined dataset
st.session_state['overall_df'] = combined_df.copy()

# Main application title and introduction
st.title("Bolts Post-Match Review App")
st.markdown("Select the Team, Opponent, and Date (Optional) to See the Post-Match Review")

# Selecting the Bolts team
teams = list(combined_df['Team Name'].unique())

selected_team = st.session_state.get('selected_team', teams[0])
if selected_team not in teams:
    selected_team = teams[0]  # Default to the first date if not found

selected_team = st.selectbox('Choose the Bolts Team:', teams, index=teams.index(selected_team))
st.session_state['selected_team'] = selected_team

# Filtering based on the selected team
combined_df = combined_df.loc[combined_df['Team Name'] == st.session_state['selected_team']]

# Selecting the opponent team
opps = list(combined_df['Opposition'].unique())

selected_opp = st.session_state.get('selected_opp', opps[0])
if selected_opp not in opps:
    selected_opp = opps[0]  # Default to the first date if not found
selected_opp = st.selectbox('Choose the Opposition:', opps, index=opps.index(selected_opp))
st.session_state['selected_opp'] = selected_opp

# Filtering based on the selected opponent
combined_df = combined_df.loc[combined_df['Opposition'] == st.session_state['selected_opp']]

# Selecting the date
dates = list(combined_df['Date'].unique())

# Check if the selected date in the session state exists in the list of dates
selected_date = st.session_state.get('selected_date', dates[0])
if selected_date not in dates:
    selected_date = dates[0]  # Default to the first date if not found

# Create the selectbox for the date
selected_date = st.selectbox('Choose the Date (if necessary)', dates, index=dates.index(selected_date))
st.session_state['selected_date'] = selected_date

# Filtering based on the selected date
combined_df = combined_df.loc[combined_df['Date'] == st.session_state['selected_date']]

# Storing the filtered DataFrame in session state
st.session_state["combined_df"] = combined_df
st.session_state['combined_df_copy'] = combined_df.copy()

# TEMPORARY: Goalkeeper-specific DataFrame for current selection
gk_dataframe = combined_df.loc[combined_df['Position Tag'] == 'GK'].reset_index(drop=True)
st.session_state['gk_df'] = gk_dataframe

# Display any relevant information or widgets below this line

