import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

# Define file paths for saving/loading models and median
LOG_REG_MODEL_PATH = 'models/log_reg_classifier_model.joblib' # Renamed for clarity
XGB_REGRESSOR_MODEL_PATH = 'models/xgb_regressor_model.joblib' # New path for XGBoost Regressor
MEDIAN_YPG_PATH = 'models/median_nfl_ypg.pkl' # Still needed for Logistic Regression's binarization

def receiverLinearPrediction():
    #Here is where I put it
    df = df[df['nfl_games'] > 4]
    df = pd.read_csv('models/combined_df.csv')
    test_df = pd.read_csv('models/merged_df.csv')
    """
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
    """
    # Prepare the features and target variable
    x = df[['YPG', 'Yards_percentage', 'draft_pick', 'RPG', 'TDPG', 
            'second_last_TDPG', 'second_last_RPG', 'second_last_YPG', 
            'conference_SEC', 'conference_ACC', 'conference_Big Ten', 
            'conference_Big 12']]
    y = df['nfl_YPG']
    # Ensure all values are numeric
    x = x.apply(pd.to_numeric, errors='coerce')         
    y = pd.to_numeric(y, errors='coerce')
    # Drop rows with missing values
    data = pd.concat([x, y], axis=1).dropna()       
    x = data[x.columns]
    y = data['nfl_YPG']
    #Now train our model and split our set into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29)       
    # Train the model   
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=25, learning_rate=0.2)
    model.fit(x_train, y_train)    
    # Show feature importance
    xgb.plot_importance(model)
    plt.show()
    #make prediction on the x_test set
    train_predictions = model.predict(x_test)

    results_df = df.loc[x_test.index, ['name']].copy()
    results_df['actual_nfl_YPG'] = y_test.values # Use y_test, not y, as y_test is the actual values for x_test
    results_df['predicted_nfl_YPG'] = train_predictions

    results_df.sort_values(by='predicted_nfl_YPG', ascending=False, inplace=True)
    print(results_df[['name', 'actual_nfl_YPG', 'predicted_nfl_YPG']].head(35))

    # Print out our predictions and show in dataframe called results_df
    #print prediction by player name
    # results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    # results_df.sort_values(by='Predicted', ascending=False, inplace=True) 
    # results_df['name'] = x_test.index
    # results_df = results_df[['name', 'Actual', 'Predicted']]      
    # print(results_df)
    #print mse
    mse = mean_squared_error(y_test, train_predictions)
    print(f'Mean Squared Error: {mse}')
    

# Define base directory for models
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure the models directory exists

# Define target columns for prediction
TARGET_COLUMNS = ['nfl_RPG', 'nfl_YPG', 'nfl_TDPG']

# Define the minimum games played threshold for filtering the training data
MIN_NFL_GAMES_THRESHOLD = 4 # You can adjust this value (e.g., to 8)

def _train_single_stacked_model(target_column, x_cleaned, y_continuous_cleaned, names_cleaned, random_state=42):
    """
    Helper function to train a single stacked model for a given target column.
    Returns the trained Logistic Regression model, XGBoost Regressor model, and median threshold.
    """
    print(f"\n--- Training Stacked Model for Target: {target_column} ---")

    # Binarize the target variable for Logistic Regression (Stage 1)
    # Median is calculated from the training split of the cleaned continuous target
    print(f"Binarizing target variable '{target_column}' for Logistic Regression (above/below median)...")
    
    # Perform initial split to get a training set for median calculation (avoid data leakage)
    # This split is temporary to calculate the median from the training portion only.
    _, _, y_train_temp_median, _ = train_test_split(x_cleaned, y_continuous_cleaned, test_size=0.2, random_state=random_state)
    median_value = y_train_temp_median.median()
    print(f"Median {target_column} (from training data) used for binarization: {median_value:.2f}")

    y_binary = (y_continuous_cleaned > median_value).astype(int)

    # Perform train-test split for both stages.
    # It's crucial that x_train/x_test for the regressor match the data used for log_reg_probs.
    x_train, x_test, y_train_continuous, y_test_continuous, \
    y_train_binary, y_test_binary, names_train, names_test = \
        train_test_split(x_cleaned, y_continuous_cleaned, y_binary, names_cleaned,
                         test_size=0.2, random_state=random_state)

    # --- Stage 1: Logistic Regression (Classification) ---
    print(f"\nStage 1: Training Logistic Regression (Classifier) for {target_column}")
    log_reg_model = LogisticRegression(random_state=random_state, solver='liblinear', max_iter=1000)
    log_reg_model.fit(x_train, y_train_binary)

    # Get predicted probabilities for the positive class from Logistic Regression
    # These probabilities will be a new feature for the XGBoost Regressor
    log_reg_probs_train = log_reg_model.predict_proba(x_train)[:, 1]
    log_reg_probs_test = log_reg_model.predict_proba(x_test)[:, 1]
    
    # Optional: Print LR classification metrics
    # log_reg_preds_test = log_reg_model.predict(x_test)
    # print(f"Logistic Regression Evaluation for {target_column} (on test set, binary target):")
    # print(f"Accuracy: {accuracy_score(y_test_binary, log_reg_preds_test):.4f}")
    # print("Classification Report:\n", classification_report(y_test_binary, log_reg_preds_test))
    # print("Confusion Matrix:\n", confusion_matrix(y_test_binary, log_reg_preds_test))

    # --- Stage 2: XGBoost Regressor (Regression) ---
    print(f"\nStage 2: Training XGBoost Regressor (Meta-Model) for {target_column}")
    # Combine original features with Logistic Regression probabilities for the meta-model
    x_train_meta = x_train.copy()
    x_train_meta['log_reg_prob'] = log_reg_probs_train.tolist()

    x_test_meta = x_test.copy()
    x_test_meta['log_reg_prob'] = log_reg_probs_test.tolist()

    # Train XGBoost Regressor on the combined features and the original continuous target
    xgb_regressor_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=random_state)
    xgb_regressor_model.fit(x_train_meta, y_train_continuous)

    # Make final continuous predictions with the XGBoost Regressor
    final_predictions_continuous = xgb_regressor_model.predict(x_test_meta)

    print(f"\nStacked Model ({target_column}) Evaluation (on test set):")
    mse = mean_squared_error(y_test_continuous, final_predictions_continuous)
    r2 = r2_score(y_test_continuous, final_predictions_continuous)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Create a DataFrame to show actual vs predicted for the test set
    results_df = pd.DataFrame({
        'name': names_test,
        f'actual_{target_column}': y_test_continuous,
        f'predicted_{target_column}': final_predictions_continuous
    })

    # Sort by predicted value for review
    results_df.sort_values(by=f'predicted_{target_column}', ascending=False, inplace=True)

    # Print the top 35 results in the requested format
    print(f"\nTop 35 Players by Stacked Model Predicted {target_column}:")
    print(results_df[['name', f'actual_{target_column}', f'predicted_{target_column}']].head(35))

    return log_reg_model, xgb_regressor_model, median_value

def train_stacked_model():
    """
    Trains stacked models for multiple target variables: nfl_YPG, nfl_recs, nfl_recYDS, nfl_recTDS.
    Saves the trained models and median YPG for each target for later prediction.
    Includes filtering for minimum NFL games played in the training data.
    """
    print("Loading data for training...")
    try:
        df = pd.read_csv(os.path.join(MODELS_DIR, 'combined_df.csv'))
    except FileNotFoundError:
        print(f"Error: '{os.path.join(MODELS_DIR, 'combined_df.csv')}' not found. Please ensure the file exists.")
        return

    # Define original features
    features = ['YPG', 'Yards_percentage', 'draft_pick', 'RPG', 'TDPG',
                'second_last_TDPG', 'second_last_RPG', 'second_last_YPG',
                'conference_SEC', 'conference_ACC', 'conference_Big Ten',
                'conference_Big 12']

    # Ensure all values are numeric and handle missing values consistently across all relevant columns
    print("Preprocessing training data (converting to numeric and handling NaNs)...")
    
    # Select all features and all potential target columns, plus 'name' and 'nfl_games' for cleaning
    all_cols_for_cleaning = features + TARGET_COLUMNS + ['name', 'nfl_games']
    data_for_cleaning_raw = df[all_cols_for_cleaning].copy()

    # Convert all relevant columns to numeric, coercing errors
    for col in features + TARGET_COLUMNS + ['nfl_games']:
        data_for_cleaning_raw[col] = pd.to_numeric(data_for_cleaning_raw[col], errors='coerce')

    # Drop rows with any missing values in features or target columns
    data_for_cleaning = data_for_cleaning_raw.dropna()

    # --- Filtering Step for training data ---
    initial_rows = len(data_for_cleaning)
    data_for_cleaning = data_for_cleaning[data_for_cleaning['nfl_games'] >= MIN_NFL_GAMES_THRESHOLD]
    filtered_rows = len(data_for_cleaning)
    print(f"Filtered out {initial_rows - filtered_rows} rows with less than {MIN_NFL_GAMES_THRESHOLD} NFL games played.")
    # --- End Filtering Step ---

    # Re-extract cleaned features and names after filtering, to be used across all target trainings
    x_cleaned_common = data_for_cleaning[features]
    names_cleaned_common = data_for_cleaning['name']

    trained_models = {}
    for target_col in TARGET_COLUMNS:
        y_continuous_current_target = data_for_cleaning[target_col]

        log_reg_model, xgb_regressor_model, median_value = \
            _train_single_stacked_model(target_col, x_cleaned_common, y_continuous_current_target, names_cleaned_common)

        # Save the trained models and median for the current target
        joblib.dump(log_reg_model, os.path.join(MODELS_DIR, f'log_reg_classifier_model_{target_col}.joblib'))
        joblib.dump(xgb_regressor_model, os.path.join(MODELS_DIR, f'xgb_regressor_model_{target_col}.joblib'))
        joblib.dump(median_value, os.path.join(MODELS_DIR, f'median_{target_col}.pkl'))
        print(f"Models and median for {target_col} saved successfully.")
        trained_models[target_col] = {
            'log_reg': log_reg_model,
            'xgb_regressor': xgb_regressor_model,
            'median': median_value
        }
    print("\nAll models trained and saved successfully.")
    return trained_models

def _predict_single_target(target_column, rookie_x_cleaned, log_reg_model, xgb_regressor_model):
    """
    Helper function to generate predictions for a single target on rookie data.
    """
    # --- Stage 1 Prediction: Logistic Regression on rookie data ---
    log_reg_probs_rookies = log_reg_model.predict_proba(rookie_x_cleaned)[:, 1]

    # --- Stage 2 Prediction: XGBoost Regressor on combined features ---
    rookie_x_meta = rookie_x_cleaned.copy()
    rookie_x_meta['log_reg_prob'] = log_reg_probs_rookies.tolist()

    final_predictions_continuous_rookies = xgb_regressor_model.predict(rookie_x_meta)
    return final_predictions_continuous_rookies

def predict_rookies():
    """
    Loads the trained stacked models for all targets and makes predictions on new rookie data
    from 'models/merged_df.csv'. Writes all predictions to a single CSV file.
    """
    print("\n--- Generating predictions for rookies using the stacked models ---")
    
    # Load all trained models and medians
    loaded_models = {}
    for target_col in TARGET_COLUMNS:
        try:
            log_reg_model = joblib.load(os.path.join(MODELS_DIR, f'log_reg_classifier_model_{target_col}.joblib'))
            xgb_regressor_model = joblib.load(os.path.join(MODELS_DIR, f'xgb_regressor_model_{target_col}.joblib'))
            median_value = joblib.load(os.path.join(MODELS_DIR, f'median_{target_col}.pkl'))
            loaded_models[target_col] = {
                'log_reg': log_reg_model,
                'xgb_regressor': xgb_regressor_model,
                'median': median_value
            }
            print(f"Models for {target_col} loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Models for {target_col} not found. Please run train_stacked_model() first.")
            return

    try:
        rookie_df = pd.read_csv(os.path.join(MODELS_DIR, 'merged_df.csv'))
    except FileNotFoundError:
        print(f"Error: '{os.path.join(MODELS_DIR, 'merged_df.csv')}' not found. Please ensure the file exists.")
        return

    # Define features (must match training features)
    features = ['YPG', 'Yards_percentage', 'draft_pick', 'RPG', 'TDPG',
                'second_last_TDPG', 'second_last_RPG', 'second_last_YPG',
                'conference_SEC', 'conference_ACC', 'conference_Big Ten',
                'conference_Big 12']

    # Extract features and names from rookie data
    rookie_x_original = rookie_df[features].copy() # Use .copy() to avoid SettingWithCopyWarning
    rookie_names = rookie_df['name'].copy()

    # Preprocessing for rookie data (must match training preprocessing)
    print("Preprocessing rookie data...")
    for col in features:
        rookie_x_original[col] = pd.to_numeric(rookie_x_original[col], errors='coerce')

    # Handle NaNs in rookie data: drop rows with missing feature values
    # IMPORTANT: Do NOT filter by 'nfl_games' here, as rookies haven't played NFL games yet.
    rookie_data_cleaned = pd.concat([rookie_x_original, rookie_names], axis=1).dropna()

    rookie_x_cleaned = rookie_data_cleaned[features]
    rookie_names_cleaned = rookie_data_cleaned['name']

    if rookie_x_cleaned.empty:
        print("No valid rookie data after cleaning. Cannot make predictions.")
        return

    # Initialize DataFrame for all rookie predictions
    rookie_predictions_df = pd.DataFrame({'name': rookie_names_cleaned})

    # Generate predictions for each target column
    for target_col in TARGET_COLUMNS:
        print(f"Generating predictions for {target_col}...")
        current_models = loaded_models[target_col]
        predictions = _predict_single_target(
            target_col,
            rookie_x_cleaned,
            current_models['log_reg'],
            current_models['xgb_regressor']
        )
        rookie_predictions_df[f'predicted_{target_col}'] = predictions

    # Sort by predicted_nfl_YPG for primary review
    if 'predicted_nfl_YPG' in rookie_predictions_df.columns:
        rookie_predictions_df.sort_values(by='predicted_nfl_YPG', ascending=False, inplace=True)
    else: # Fallback if nfl_YPG is not a target for some reason
        rookie_predictions_df.sort_values(by=rookie_predictions_df.columns[1], ascending=False, inplace=True)


    print("\nAll Rookie Predictions:")
    print(rookie_predictions_df.head(35)) # Display all or top 35 rookies

    # Save predictions to CSV
    output_csv_path = os.path.join(MODELS_DIR, 'rookie_predictions.csv')
    rookie_predictions_df.to_csv(output_csv_path, index=False)
    print(f"\nRookie predictions saved to '{output_csv_path}'")

    print("\nRookie prediction process completed.")


if __name__ == '__main__':
    # receiverLinearPrediction()
    train_stacked_model()
    predict_rookies()
    # print("Receiver Linear Prediction completed.")
    # print("Now running XGBoost prediction...")
    # xgboostPrediction()
    # print("XGBoost prediction completed.")    