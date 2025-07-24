import pandas as pd

def combine_data():
    df  = pd.read_csv('models/hi.csv')
    alterSchools(df)
    print(df.head(10))
    

    #College to conference mapping
def alterSchools(df: pd.DataFrame):
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
            "Ohio St.": "Big Ten",
            "Penn St.": "Big Ten",
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
            "NC St.": "ACC",
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
            "Iowa St.": "Big 12",
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
    df.to_csv('test_set.csv', index=False)
    print(df.head())

if __name__ == '__main__':
    combine_data()
   