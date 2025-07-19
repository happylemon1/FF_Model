import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import pandas as pd
import matplotlib.pyplot as plt

def receiverLinearPrediction():
    df = pd.read_csv('combined_receiver_data2.csv')
    # print(df.head())

    #edit the test df for use
    test_df = pd.read_csv('models/hi.csv')
    test_df['name'] = test_df['name'].str.strip()
    # print(test_df.columns)
    # print(test_df.head(10))

    #split by season
    df_24 = test_df[test_df['season'] == 2024]
    df_23 = test_df[test_df['season'] == 2023]
    # Add 24 per game averages
    df_24['games'] = (df_24['recYDS']/ df_24['YPG'])
    print("Games", df_24['games'])
    df_24['RPG'] = df_24['recs']/ df_24['games']
    df_24['TDPG'] = df_24['recTDS']/ df_24['games']
    df_24['Yards_percentage'] = df_24['recYDS'] / df_24['TEAM_YARDS']
    df_23['games'] = (df_23['recYDS']/ df_23['YPG'])
    df_23['second_last_TDPG'] = df_23['recTDS']/ df_23['games']
    df_23['second_last_RPG'] = df_23['recs']/ df_23['games']
    df_23 = df_23.rename( columns = {
        'YPG': 'second_last_YPG'
    })

    # print(df_24.head(10))
    # print(df_23.head())
    merged_df = pd.merge(df_24, df_23[['name', 'second_last_TDPG', 'second_last_RPG', 'second_last_YPG']], on='name', how='left')
    # print("Merged df", merged_df.head(3))

    # Get the actual numeric data from the columns
    x = df[['YPG', 'Yards_percentage', 'draft_pick', 'RPG', 'TDPG', 'second_last_TDPG', 'second_last_RPG', 'second_last_YPG']]
    y = df['nfl_YPG']

    # Make sure all values are numeric
    x = x.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with missing values
    data = pd.concat([x, y], axis=1).dropna()
    x = data[x.columns]
    y = data['nfl_YPG']

    # Train the model
    model = LinearRegression()
    model.fit(x, y)

    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print('this method ran ')

    #Now we do the model predictions
    features= ['YPG', 'Yards_percentage', 'draft_pick', 'RPG', 'TDPG', 'second_last_TDPG', 'second_last_RPG', 'second_last_YPG']

    featured_df = merged_df[features]
    # print("Featured df: ", featured_df)
    featured_df = featured_df.apply(pd.to_numeric, errors='coerce')
    # print(featured_df.head(1))
    predictions = model.predict(featured_df)
    merged_df['predicted_nfl_YPG'] = predictions
    print(merged_df[['name', 'predicted_nfl_YPG']].head(20))
    

    conferences = []
    conferences.append('SEC')
    conferences.append('Big 10')
    
def main():
    receiverLinearPrediction()

if __name__  =='__main__':
    main()


