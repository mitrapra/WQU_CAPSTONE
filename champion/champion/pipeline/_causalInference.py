import time  # Importing the time module to measure execution time

from champion.source.models.Logistic_Regression import Logistic_Regression  # Importing Logistic Regression model
from champion.source.models.Ensemble_Stacking import Ensemble_Stacking  # Importing Ensemble Stacking model
from champion.source.models.Ensemble_Voting import Ensemble_Voting  # Importing Ensemble Voting model
from champion.source.models.RandomForest import RandomForest  # Importing Random Forest model
from champion.source.models.LongShortTM import LongShortTM  # Importing Long Short-Term Memory (LSTM) model for sequential data
from champion.source.models.LightGBM import LightGBM  # Importing LightGBM model
from champion.source.models.XGBoost import XGBoost  # Importing XGBoost model
from champion.source.models.ANN import ANN  # Importing Artificial Neural Network (ANN) model

if __name__ == "__main__":
    # Measure the execution time for each model's SHAP value generation

    # 1. Logistic Regression
    start = time.time()  # Start the timer
    log_model = Logistic_Regression()  # Instantiate the Logistic Regression model
    log_model.generate_shap_value()  # Generate SHAP values for the Logistic Regression model
    log_train_time = time.time() - start  # Calculate the execution time

    # 3. Random-Forest
    start = time.time()  # Start the timer
    rf_model = RandomForest()  # Instantiate the Random Forest model
    rf_model.generate_shap_value()  # Generate SHAP values for the Random Forest model
    rf_train_time = time.time() - start  # Calculate the execution time

    # 4. Boosting
    start = time.time()  # Start the timer
    lightgbm_model = LightGBM()  # Instantiate the LightGBM model
    lightgbm_model.generate_shap_value()  # Generate SHAP values for the LightGBM model
    lightgbm_train_time = time.time() - start  # Calculate the execution time

    start = time.time()  # Start the timer
    xgboost_model = XGBoost()  # Instantiate the XGBoost model
    xgboost_model.generate_shap_value()  # Generate SHAP values for the XGBoost model
    xgboost_train_time = time.time() - start  # Calculate the execution time

    # 5. Neural Network - ANN
    start = time.time()  # Start the timer
    ann_model = ANN()  # Instantiate the ANN model
    ann_model.generate_shap_value()  # Generate SHAP values for the ANN model
    ann_train_time = time.time() - start  # Calculate the execution time

    # 6. Neural Network - LSTM
    start = time.time()  # Start the timer
    lstm_model = LongShortTM()  # Instantiate the LSTM model
    lstm_model.generate_shap_value()  # Generate SHAP values for the LSTM model
    lstm_train_time = time.time() - start  # Calculate the execution time

    # 7. Ensembly of Models
    start = time.time()  # Start the timer
    ensemble_voting_model = Ensemble_Voting()  # Instantiate the Ensemble Voting model
    ensemble_voting_model.generate_shap_value()  # Generate SHAP values for the Ensemble Voting model
    voting_train_time = time.time() - start  # Calculate the execution time

    start = time.time()  # Start the timer
    ensemble_stacking_model = Ensemble_Stacking()  # Instantiate the Ensemble Stacking model
    ensemble_stacking_model.generate_shap_value()  # Generate SHAP values for the Ensemble Stacking model
    stacking_train_time = time.time() - start  # Calculate the execution time

    # Print the execution times for SHAP value generation for each model
    print(f"""
    The time taken (s) to generate SHAP Value for the following are noted below:
    Logistic        : {log_train_time:.2f}
    Random Forest   : {rf_train_time:.2f}
    LightGBM        : {lightgbm_train_time:.2f}
    XGBoost         : {xgboost_train_time:.2f}
    ANN             : {ann_train_time:.2f}
    LSTM            : {lstm_train_time:.2f}
    Voting          : {voting_train_time:.2f}
    Stacking        : {stacking_train_time:.2f}
    """)