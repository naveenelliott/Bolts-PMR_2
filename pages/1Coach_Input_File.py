import streamlit as st
import pandas as pd

# Getting the selected team, opponent, and date
selected_team = st.session_state["selected_team"]
selected_opp = st.session_state["selected_opp"]
selected_date = st.session_state["selected_date"]

st.set_page_config(layout='wide')


# function that will save the input
def save_input(in_possession, out_possession):

    overall = st.session_state['overall_df']
    overall.loc[overall['Player Full Name'] == 'Casey Powers', 'Position Tag'] = 'GK'
    condition = (
        (overall['Team Name'] == st.session_state['selected_team']) &
        (overall['Opposition'] == st.session_state['selected_opp']) &
        (overall['Date'] == st.session_state['selected_date'])
    )
    overall.loc[condition, 'In Possession'] = in_possession
    overall.loc[condition, 'Out Possession'] = out_possession
    limited_df = overall[['Team Name', 'Date', 'Opposition', 'In Possession', 'Out Possession']]
    limited_df.to_csv('pages/InAndOutOfPossessionGoals.csv', index=False)
    st.session_state['overall_df'] = overall

def main():
    st.title("Setting In and Out of Possession Goals")

    st.markdown(f"<h3 style='text-align: center;'>Team: {selected_team}&nbsp;|&nbsp;Opposition: {selected_opp}</h3>", unsafe_allow_html=True)

    # Display current DataFrame
    overall = st.session_state['overall_df']
    condition = (
        (overall['Team Name'] == st.session_state['selected_team']) &
        (overall['Opposition'] == st.session_state['selected_opp']) &
        (overall['Date'] == st.session_state['selected_date'])
    )
    overall = overall.loc[condition]
    current_in_possession = overall.iloc[0]['In Possession']
    current_out_possession = overall.iloc[0]['Out Possession']
    if pd.isna(current_in_possession):
        current_in_possession = 'Nothing, needs updated.'
    if pd.isna(current_out_possession):
        current_out_possession = 'Nothing, needs updated.'

    st.write(f"In Possession Current Goals: {current_in_possession}")
    st.write(f"Out Possession Current Goals: {current_out_possession}")

    # Form to update the DataFrame
    with st.form("input_form"):
        in_possession = st.text_input("In Possession:")
        out_possession = st.text_input("Out of Possession:")
        submit_button = st.form_submit_button(label='Save')

        if submit_button:
            save_input(in_possession, out_possession)
            st.success("Input updated!")
            st.experimental_rerun()  # Rerun to refresh the displayed DataFrame

if __name__ == "__main__":
    main()

