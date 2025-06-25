
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
    college_df['TDPG'] = college_df['recTDs'] / college_df['games']
    college_df['RPG'] = college_df['recs']/ college_df['games']
    college_df['season'] = college_df['season'].str.extract(r'(\d{4})').astype(int)
    college_df = college_df[college_df['school'].notna()]
    #For each player add how their college did that season in terms of receiving statistics
    college_df = addCollegiateInfo(college_df)


    # Calculate the trending average of the last two seasons to see which direction the receiver is going to
    grouped = college_df.groupby('name')

    for player_name, group in grouped:

        # this trend df should hold all the rows that we can append for our new dataframe used in our model
        trend_dataframe = []
        #setting each playe group as a temporary dataframe
        temp_df = group
        lastSeason = temp_df.iloc[-1]
        secondLast = temp_df.iloc[-2]
        #diff values
        YPG_diff = lastSeason['YPG'] - secondLast['YPG']
        RPG_diff = lastSeason['RPG'] - secondLast['RPG']
        TDPG_diff = lastSeason['TDPG'] - secondLast['TDPG']
        #Row that we are going to append
        last = lastSeason.copy()
        #setting the new parameters into lastSeasn
        lastSeason['YPG_diff'] = YPG_diff
        lastSeason['RPG_diff'] = RPG_diff
        lastSeason['TDPG_diff'] = TDPG_diff
        lastSeason['second_last_TDPG'] = secondLast['TDPG']
        lastSeason['second_last_RPG'] = secondLast['RPG']
        lastSeason['second_last_YPG'] = secondLast['YPG']
        #append to data frame


def addCollegiateInfo(college_df):
    for index, row in college_df.itterows():
        schoolName = str(row['school'])
        year = str(row['season'])

        #Here we try to create the portions of the URL
        parts = schoolName.split(' ')

        if parts[-1] == 'st.':
            parts[-1] = 'state'

        #No dashes are needed if there is only one part
        if parts.length == 1:
            collegeURL = 'https://www.sports-reference.com/cfb/schools/' + str(parts[0]) + '/{season}.html'
        else :
            semiURL = ''
            for i in range(len(parts) - 1):
                semiURL = semiURL + str(parts[i]) + '-'
            semiURL + str(parts[-1])
            collegeURL = 'https://www.sports-reference.com/cfb/schools/' + semiURL + '/{season}.html'
        
        print(collegeURL)
        #From here create a new response
        time.sleep(2)
        response = requests.get(collegeURL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        #Find the specific part of the html that we need
        table = soup.find('table', {'id':'index'})
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
        
            





            

            





