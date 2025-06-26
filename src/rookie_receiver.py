
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Comment
import requests
import time

def load_and_clean_receiver_data(college_data_path, nfl_data_path):


    try: 
        college_df = pd.read_csv(college_data_path)
        nfl_df = pd.read_csv(nfl_data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found. Please check paths.")
        return None

    # Creating some new columns for per game basis in case players missed games
    college_df = college_df.drop(columns = ['age'])
    college_df['YPC'] = college_df['recYDS'] / college_df['recs']
    college_df ['YPG'] = college_df['recYDS'] / college_df['games']
    college_df['TDPG'] = college_df['recTDS'] / college_df['games']
    college_df['RPG'] = college_df['recs']/ college_df['games']
    college_df['season'] = college_df['season'].str.extract(r'(\d{4})').astype(int)
    college_df = college_df[college_df['school'].notna()]
    #For each player add how their college did that season in terms of receiving statistics
    #college_df = addCollegiateInfo(college_df)


    # Calculate the trending average of the last two seasons to see which direction the receiver is going to
    grouped = college_df.groupby('name')
    trend_data_list = []
    for player_name, group in grouped:
        if (len(group) >= 2):
            group = group.sort_values('season').copy()

            # this trend df should hold all the rows that we can append for our new dataframe used in our model
            
            #setting each playe group as a temporary dataframe
            temp_df = group
            lastSeason = temp_df.iloc[-1].copy()
            secondLast = temp_df.iloc[-2].copy()
            #diff values
            YPG_diff = lastSeason['YPG'] - secondLast['YPG']
            RPG_diff = lastSeason['RPG'] - secondLast['RPG']
            TDPG_diff = lastSeason['TDPG'] - secondLast['TDPG']
            
            #setting the new parameters into lastSeasn
            lastSeason['YPG_diff'] = YPG_diff
            lastSeason['RPG_diff'] = RPG_diff
            lastSeason['TDPG_diff'] = TDPG_diff
            lastSeason['second_last_TDPG'] = secondLast['TDPG']
            lastSeason['second_last_RPG'] = secondLast['RPG']
            lastSeason['second_last_YPG'] = secondLast['YPG']
        #append to data frame
            trend_data_list.append(lastSeason)
    if trend_data_list:
        final_college_df = pd.DataFrame(trend_data_list)
    else:
        final_college_df = pd.DataFrame(columns=college_df.columns.tolist() + ['YPG_diff', 'RPG_diff', 'TDPG_diff', 'second_last_TDPG', 'second_last_RPG', 'second_last_YPG'])

    print(final_college_df.head(9))


def addCollegiateInfo(college_df):
    for index, row in college_df.iterrows():
        semiURL = college_df['collegeURL']
        
        realURL = 'https://www.sports-reference.com' + str(semiURL)
        #From here create a new response
        time.sleep(2)
        response = requests.get(realURL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        #Find the specific part of the html that we need
        table = soup.find('table', {'id':'team'})
        tbody = table.find('tbody')
        offense_row = tbody.find('tr')
        #Get the teams, YPG, TDPG, and completions per game (RPG).
        teamRPG = float(offense_row.find('td', {'data-stat': 'pass_cmp'}).text.strip())
        print(teamRPG)
        teamYPG = float(offense_row.find('td', {'data-stat': 'pass_yds'}).text.strip())
        print(teamYPG)
        teamTDPG = float(offense_row.find('td', {'data-stat':'pass_td'}).text.strip())
        print(teamTDPG)
        #Assign these values back into each row
        college_df.loc[index, 'Team_RPG'] = teamRPG
        college_df.loc[index, 'Team_YPG'] = teamYPG
        college_df.loc[index, 'Team_TDPG'] = teamTDPG

    return college_df
        
            





            

            





