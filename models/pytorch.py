import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

def receiverLinearPrediction():
    df = pd.read_csv('combined_receiver_data2.csv')
    print(df.head())



    # Get the actual numeric data from the columns
    X = df[['draft_pick', 'YPG', 'RPG', 'TDPG', 'age', 'Yards_percentage', 'second_last_YPG', 'second_last_RPG', 'second_last_TDPG']]
    y = df['nfl_YPG']

    # Make sure all values are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with missing values
    data = pd.concat([X, y], axis=1).dropna()
    X = data[X.columns]
    y = data['nfl_YPG']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print('this method ran ')

    testDataset = pd.read_csv()

def main():
    receiverLinearPrediction()

if __name__  =='__main__':
    main()


