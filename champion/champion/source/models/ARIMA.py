import pmdarima as pm
import pandas as pd
import numpy as np
import pickle
import joblib
import os

from champion.config import _paths as p
from champion.source.util import _util as util
from champion.source.models._mainModeling import _mainModeling
from champion.source.model_evaluation._evaluation_metrics import aic_calc, bic_calc, classification_score

class ARIMA(_mainModeling):
    """A class for training and evaluating an ARIMA model."""

    def __init__(self):
        """Initialize the ARIMA class."""
        super().__init__()
        self.curr_config = self.config.get("ARIMA")
        self.model_name  = self.curr_config.get("name")

    def data_preprocessing(self):
        """Preprocess the data."""
        y_column            = self.curr_config.get("time-series")
        x_column            = self.curr_config.get("exogenous")
        time_series_data_train = np.asarray(self.train_set[y_column]).reshape(-1, 1)
        exogenous_data_train   = np.asarray(self.train_set[x_column]).reshape(-1, 1)
        exogenous_data_test    = np.asarray(self.test_set[x_column]).reshape(-1, 1)
        return time_series_data_train, exogenous_data_train, exogenous_data_test

    def model_training(self):
        """Train the ARIMA model."""
        time_series_data, exogenous_data_train, exogenous_data_test = self.data_preprocessing()
        self.model = pm.auto_arima(
            time_series_data, 
            exogenous=exogenous_data_train,
            start_p=1, 
            start_q=1,
            test='adf',
            max_p=6, 
            max_q=6, 
            m=1,
            start_P=0, 
            seasonal=True,
            d=None, 
            D=0, 
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        # generate forecast
        y_pred, confidence_interval = self.model.predict(n_periods=self.test_set.shape[0],
                                                         exogenous=exogenous_data_test,
                                                         return_conf_int=True)

        # transform the forecast into price signals
        y_pred_df 				= pd.DataFrame(y_pred, columns=["forecast"], index=self.test_set["Date"])
        benchmark 				= self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["forecast"].apply(lambda x: int(x >= benchmark))

        # output the prediction
        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))

    def model_evaluation(self):
        """Evaluate the ARIMA model.""" 
        self.model       = joblib.load(os.path.join(p.model_path, f"{self.model_name}.pkl"))
        time_series_data, exogenous_data_train, exogenous_data_test = self.data_preprocessing()
        benchmark        = self.curr_config.get("prediction_benchmark")

        # 1. evaluate model performance on train set
        train_predicted  = self.model.arima_res_.fittedvalues
        aic_train        = self.model.aic()
        bic_train        = self.model.bic()
        train_prediction = np.vectorize(util.map_class)(train_predicted, benchmark)
        res_train, acc_train, precision_train, recall_train, f1_train, cm_train = classification_score(train_prediction,
                                                                                                self.train_set["target"],0)

        train_report = pd.DataFrame({
            self.model_name: {
                "Train_Accuracy" : acc_train,
                "Train_Precision": precision_train,
                "Train_Recall"   : recall_train,
                "Train_F1-Score" : f1_train,
                "Train_AIC"      : aic_train,
                "Train_BIC"      : bic_train
            }
        }).fillna(float(0))
        train_report.to_csv(os.path.join(p.model_evaluation_report_path, f"{self.model_name}_train_report.csv"))

        # evaluate model performance on test set
        y_pred, confidence_interval = self.model.predict(n_periods=self.test_set.shape[0],
                                                         exogenous=exogenous_data_test,
                                                         return_conf_int=True)
        y_pred_df                   = pd.DataFrame(y_pred, columns=["forecast"], index=self.test_set["Date"])
        y_pred_df["final_pred"]     = y_pred_df["forecast"].apply(lambda x: int(x >= benchmark))
        y_column                    = self.curr_config.get("time-series")
        k                           = self.model.params().shape[0]
        aic_test                    = aic_calc(self.test_set[y_column], y_pred_df["forecast"], k, "regression")
        bic_test                    = bic_calc(self.test_set[y_column], y_pred_df["forecast"], k, "regression")
        res_test, acc_test, precision_test, recall_test, f1_test, cm_test = classification_score(y_pred_df["final_pred"],
                                                                                                self.test_set["target"],
                                                                                                0)

        test_report = pd.DataFrame({
            self.model_name: {
                "Test_Accuracy" : acc_test,
                "Test_Precision": precision_test,
                "Test_Recall"   : recall_test,
                "Test_F1-Score" : f1_test,
                "Test_AIC"      : aic_test,
                "Test_BIC"      : bic_test
            }
        }).fillna(float(0))
        test_report.to_csv(os.path.join(p.model_evaluation_report_path, f"{self.model_name}_test_report.csv"))
    
    def backtest_evaluation(self, backtest: str = "stress"):
        self.model       = joblib.load(os.path.join(p.model_path, f"{self.model_name}.pkl"))
        # Perform evaluation on stress data.
        stress_data_path = os.path.join(p.market_data_output_folder, "stress_period.csv")
        stress_data 	 = pd.read_csv(stress_data_path)
        benchmark        = self.curr_config.get("prediction_benchmark")

        # evaluate model performance on stress data
        y_pred, confidence_interval = self.model.predict(n_periods=stress_data.shape[0],
                                                         exogenous=stress_data,
                                                         return_conf_int=True)
        y_pred_df                   = pd.DataFrame(y_pred, columns=["forecast"], index=stress_data["Date"])
        y_pred_df["final_pred"]     = y_pred_df["forecast"].apply(lambda x: int(x >= benchmark))
        y_column                    = self.curr_config.get("time-series")
        k                           = self.model.params().shape[0]
        aic_stress                  = aic_calc(stress_data[y_column], y_pred_df["forecast"], k, "regression")
        bic_stress                  = bic_calc(stress_data[y_column], y_pred_df["forecast"], k, "regression")
        res_stress, acc_stress, precision_stress, recall_stress, f1_stress, cm_stress = classification_score(y_pred_df["final_pred"],stress_data["target"],0)

        # Generate a report for the stress period including evaluation metrics.
        stress_report = pd.DataFrame({
            self.model_name: {
                "Test_Accuracy" : acc_stress,
                "Test_Precision": precision_stress,
                "Test_Recall"   : recall_stress,
                "Test_F1-Score" : f1_stress,
                "Test_AIC"      : aic_stress,
                "Test_BIC"      : bic_stress
            }
        }).fillna(float(0))
        stress_report.to_csv(os.path.join(p.model_evaluation_report_path, f"{self.model_name}_stress_report.csv")) 