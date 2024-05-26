import time

from champion.source.util._util import dir_management  # Importing directory management function
from champion.source.data.MarketDataEngineer import MarketDataEngineer  # Importing MarketDataEngineer class for data processing

from champion.source.models.Logistic_Regression import Logistic_Regression  # Importing Logistic Regression model
from champion.source.models.Ensemble_Stacking import Ensemble_Stacking  # Importing Ensemble Stacking model
from champion.source.models.Ensemble_Voting import Ensemble_Voting  # Importing Ensemble Voting model
from champion.source.models.RandomForest import RandomForest  # Importing Random Forest model
from champion.source.models.LongShortTM import LongShortTM  # Importing Long Short-Term Memory (LSTM) model for sequential data
from champion.source.models.LightGBM import LightGBM  # Importing LightGBM model
from champion.source.models.XGBoost import XGBoost  # Importing XGBoost model
from champion.source.models.ARIMA import ARIMA  # Importing ARIMA model for time-series forecasting
from champion.source.models.ANN import ANN  # Importing Artificial Neural Network (ANN) model

if __name__ == "__main__":
    # Ensure that the output folders are created
    dir_management()

    # Start data loading
    start = time.time()
    loader = MarketDataEngineer()
    loader.etl_process()  # Execute the ETL process
    etl_load_time = time.time() - start  # Calculate the time taken for data loading

    # 1. Logistic Regression
    start = time.time()
    log_model = Logistic_Regression()
    log_model.model_training()  # Train the Logistic Regression model
    log_model.save_model()  # Save the trained Logistic Regression model
    log_train_time = time.time() - start  # Calculate the training time for Logistic Regression

    # 2. ARIMA time-series
    start = time.time()
    arima_model = ARIMA()
    arima_model.model_training()  # Train the ARIMA model
    arima_model.save_model()  # Save the trained ARIMA model
    arima_train_time = time.time() - start  # Calculate the training time for ARIMA

    # 3. Random Forest
    start = time.time()
    rf_model = RandomForest()
    rf_model.model_training()  # Train the Random Forest model
    rf_model.save_model()  # Save the trained Random Forest model
    rf_train_time = time.time() - start  # Calculate the training time for Random Forest

    # 4. Boosting
    start = time.time()
    lightgbm_model = LightGBM()
    lightgbm_model.model_training()  # Train the LightGBM model
    lightgbm_model.save_model()  # Save the trained LightGBM model
    lightgbm_train_time = time.time() - start  # Calculate the training time for LightGBM

    start = time.time()
    xgboost_model = XGBoost()
    xgboost_model.model_training()  # Train the XGBoost model
    xgboost_model.save_model()  # Save the trained XGBoost model
    xgboost_train_time = time.time() - start  # Calculate the training time for XGBoost

    # 5. Neural Network - ANN
    start = time.time()
    ann_model = ANN()
    ann_model.model_training()  # Train the ANN model
    ann_model.save_model()  # Save the trained ANN model
    ann_train_time = time.time() - start  # Calculate the training time for ANN

    # 6. Neural Network - LSTM
    start = time.time()
    lstm_model = LongShortTM()
    lstm_model.model_training()  # Train the LSTM model
    lstm_model.save_model()  # Save the trained LSTM model
    lstm_train_time = time.time() - start  # Calculate the training time for LSTM

    # 7. Ensemble of Models
    start = time.time()
    ensemble_voting_model = Ensemble_Voting()
    ensemble_voting_model.model_training()  # Train the Ensemble Voting model
    ensemble_voting_model.save_model()  # Save the trained Ensemble Voting model
    voting_train_time = time.time() - start  # Calculate the training time for Ensemble Voting

    start = time.time()
    ensemble_stacking_model = Ensemble_Stacking()
    ensemble_stacking_model.model_training()  # Train the Ensemble Stacking model
    ensemble_stacking_model.save_model()  # Save the trained Ensemble Stacking model
    stacking_train_time = time.time() - start  # Calculate the training time for Ensemble Stacking

    # Print the time taken for each process
    print(f"""
    The time taken (s) for the following processes are noted below:
    ETL             : {etl_load_time:.2f}
    Logistic        : {log_train_time:.2f}
    ARIMA           : {arima_train_time: .2f}
    Random Forest   : {rf_train_time:.2f}
    LightGBM        : {lightgbm_train_time:.2f}
    XGBoost         : {xgboost_train_time:.2f}
    ANN             : {ann_train_time:.2f}
    LSTM            : {lstm_train_time:.2f}
    Voting          : {voting_train_time:.2f}
    Stacking        : {stacking_train_time:.2f}
    """)