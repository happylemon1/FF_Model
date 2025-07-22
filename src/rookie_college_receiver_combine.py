import pandas as pd

def combineReceiver():
    receiver_df = pd.read_csv('data/raw/wr_and_CollegeStats.csv')
    nflReceiver_df = pd.read_csv('data/raw/nfl_wr_data_2.csv')

    combineddf = pd.merge(receiver_df, nflReceiver_df, on = 'name', how = 'right')

    combineddf['Yards_percentage'] = combineddf['YPG']/ combineddf['Team_YPG']
    combineddf['Receptions_Percentage'] = combineddf['RPG']/ combineddf['Team_YPG']
    combineddf['TD_Percentage'] = combineddf['TDPG']/ combineddf['Team_TDPG']
    combineddf = combineddf.drop(columns = ['games', 'Team_YPG', 'Team_TDPG', 'Team_RPG', 'collegeURL', 'position_y', 'YPC', 'position_x'], axis = 1)
    combineddf = combineddf.dropna(axis=1, how='all')
    combineddf['nfl_YPG'] = combineddf['nfl_recYDS']/ combineddf['nfl_games']
    combineddf['nfl_TDPG'] = combineddf['nfl_recTDS']/ combineddf['nfl_games']
    combineddf['nfl_RPG'] = combineddf['nfl_recs']/ combineddf['nfl_games']

    combineddf.to_csv('combined_receiver_data2.csv', index=False)
    print(combineddf['TD_Percentage'].dtype)
    print(combineddf['recYDS'].dtype)
    alterSchools(combineddf)
    print(combineddf.head(10))

def alterSchools(df: pd.DataFrame):
    #College to conference mapping
    college_to_conference = {
    # ✅ SEC
    "Alabama": "SEC",
    "Georgia": "SEC",
    "LSU": "SEC",
    "Texas A&M": "SEC",
    "Florida": "SEC",
    "Tennessee": "SEC",
    "Auburn": "SEC",
    "Ole Miss": "SEC",
    "Mississippi St.": "SEC",
    "Arkansas": "SEC",
    "Kentucky": "SEC",
    "South Carolina": "SEC",
    "Missouri": "SEC",
    "Texas": "SEC",
    "Oklahoma": "SEC",

    # ✅ Big Ten
    "Michigan": "Big Ten",
    "Ohio State": "Big Ten",
    "Penn State": "Big Ten",
    "Michigan St.": "Big Ten",
    "Illinois": "Big Ten",
    "Indiana": "Big Ten",
    "Iowa": "Big Ten",
    "Minnesota": "Big Ten",
    "Nebraska": "Big Ten",
    "Northwestern": "Big Ten",
    "Purdue": "Big Ten",
    "Rutgers": "Big Ten",
    "Maryland": "Big Ten",
    "USC": "Big Ten",
    "UCLA": "Big Ten",
    "Oregon": "Big Ten",
    "Washington": "Big Ten",

    # ✅ ACC
    "Clemson": "ACC",
    "Florida St.": "ACC",
    "Miami (FL)": "ACC",
    "North Carolina": "ACC",
    "NC State": "ACC",
    "Duke": "ACC",
    "Wake Forest": "ACC",
    "Virginia": "ACC",
    "Virginia Tech": "ACC",
    "Georgia Tech": "ACC",
    "Syracuse": "ACC",
    "Louisville": "ACC",
    "Boston College": "ACC",
    "Pittsburgh": "ACC",
    "Cal": "ACC",
    "Stanford": "ACC",
    "SMU": "ACC",

    # ✅ Big 12
    "Baylor": "Big 12",
    "TCU": "Big 12",
    "Texas Tech": "Big 12",
    "Kansas": "Big 12",
    "Kansas St.": "Big 12",
    "Oklahoma St.": "Big 12",
    "Iowa State": "Big 12",
    "West Virginia": "Big 12",
    "Cincinnati": "Big 12",
    "Houston": "Big 12",
    "UCF": "Big 12",
    "BYU": "Big 12",
    "Arizona": "Big 12",
    "Arizona St.": "Big 12",
    "Utah": "Big 12",
    "Colorado": "Big 12"
}
    df['conference'] = df['school'].apply(lambda s: college_to_conference.get(s, 'Other'))
    df = pd.get_dummies(df, columns=['conference'])
    df.to_csv('receiver_with_conferences.csv', index=False)

    
if __name__ == '__main__':
    combineReceiver()