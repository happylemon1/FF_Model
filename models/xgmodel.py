import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def receiverLinearPrediction():
    df = pd.read_csv('models/receiver_with_conferences.csv')
    test_df = pd.read_csv('models/merged_df.csv')

    # Prepare the features and target variable
    x = df[['YPG', 'Yards_percentage', 'draft_pick', 'RPG', 'TDPG', 
            'second_last_TDPG', 'second_last_RPG', 'second_last_YPG', 
            'conference_SEC', 'conference_ACC', 'conference_Big Ten', 
            'conference_Big 12']]
    y = df['nfl_RPG']

    
    # Ensure all values are numeric
    x = x.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with missing values
    data = pd.concat([x, y], axis=1).dropna()
    x = data[x.columns]
    y = data['nfl_RPG']
    # WE have already trained model so let run it on our actual test set
    test_x = test_df[x.columns]
    test_x = test_x.apply(pd.to_numeric, errors='coerce')
    test_x = test_x.dropna(axis=1, how='all')

    #Train our Model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(x,y)
    #show feature importance
    xgb.plot_importance(model)
    plt.show()
    # Make predictions on the test set
    predictions = model.predict(test_x) 
    #Print out our predictions
    test_df['predicted_nfl_YPG'] = predictions
    test_df.sort_values(by='predicted_nfl_YPG', ascending=False, inplace=True)
    print(test_df[['name', 'predicted_nfl_YPG']].head(35))

if __name__ == '__main__':
    receiverLinearPrediction()
    # print("Receiver Linear Prediction completed.")
    # print("Now running XGBoost prediction...")
    # xgboostPrediction()
    # print("XGBoost prediction completed.")    