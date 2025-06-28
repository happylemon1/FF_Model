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
if __name__ == '__main__':
    combineReceiver()