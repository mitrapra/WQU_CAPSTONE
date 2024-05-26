# Importing necessary models for evaluation and backtesting
from champion.source.models.Logistic_Regression import Logistic_Regression  # Importing Logistic Regression model
from champion.source.models.Ensemble_Stacking import Ensemble_Stacking  # Importing Ensemble Stacking model
from champion.source.models.Ensemble_Voting import Ensemble_Voting  # Importing Ensemble Voting model
from champion.source.models.RandomForest import RandomForest  # Importing Random Forest model
from champion.source.models.LongShortTM import LongShortTM  # Importing Long Short-Term Memory (LSTM) model for sequential data
from champion.source.models.LightGBM import LightGBM  # Importing LightGBM model
from champion.source.models.XGBoost import XGBoost  # Importing XGBoost model
from champion.source.models.ARIMA import ARIMA  # Importing ARIMA model for time-series forecasting
from champion.source.models.ANN import ANN  # Importing Artificial Neural Network (ANN) model

# Importing function to compile model evaluation reports
from champion.source.model_evaluation._evaluation_metrics import compile_model_eval_reports

if __name__ == "__main__":
    # Instantiate and evaluate each model, then backtest the strategies
    log_model = Logistic_Regression()
    log_model.model_evaluation(backtest="stress")  # Evaluate the Logistic Regression model
    log_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the Logistic Regression model

    arima_model = ARIMA()
    arima_model.model_evaluation(backtest="stress")  # Evaluate the ARIMA model
    arima_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the ARIMA model

    rf_model = RandomForest()
    rf_model.model_evaluation(backtest="stress")  # Evaluate the Random Forest model
    rf_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the Random Forest model

    lightgbm_model = LightGBM()
    lightgbm_model.model_evaluation(backtest="stress")  # Evaluate the LightGBM model
    lightgbm_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the LightGBM model

    xgboost_model = XGBoost()
    xgboost_model.model_evaluation(backtest="stress")  # Evaluate the XGBoost model
    xgboost_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the XGBoost model

    ann_model = ANN()
    ann_model.model_evaluation(backtest="stress")  # Evaluate the ANN model
    ann_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the ANN model

    lstm_model = LongShortTM()
    lstm_model.model_evaluation(backtest="stress")  # Evaluate the LSTM model
    lstm_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the LSTM model

    ensemble_voting_model = Ensemble_Voting()
    ensemble_voting_model.model_evaluation(backtest="stress")  # Evaluate the Ensemble Voting model
    ensemble_voting_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the Ensemble Voting model

    ensemble_stacking_model = Ensemble_Stacking()
    ensemble_stacking_model.model_evaluation(backtest="stress")  # Evaluate the Ensemble Stacking model
    ensemble_stacking_model.backtest_strategy(backtest="stress")  # Backtest the strategy for the Ensemble Stacking model

    compile_model_eval_reports()  # Compile model evaluation reports for all models