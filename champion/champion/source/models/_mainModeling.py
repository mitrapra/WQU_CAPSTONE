# Import necessary libraries for model evaluation, data visualization, data manipulation, and file operations.
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow
import os
import pickle
import joblib
import random
import shap

from tensorflow.keras.models import load_model

# Import custom modules for evaluation metrics and utility functions.
from champion.source.model_evaluation._evaluation_metrics import aic_calc, bic_calc, classification_score, portfolio_evaluation
from champion.source.util import _util as util
from champion.config import _paths as p

# Define a class called `Modeling` to encapsulate all model-related functionalities.
class _mainModeling:

    # Initialize the class with necessary attributes and configurations.
    def __init__(self):
        # Load training and testing datasets from CSV files defined in the paths module.
        self.train_set = pd.read_csv(p.train_set_path)
        self.test_set  = pd.read_csv(p.test_set_path)
        
        # Load model configuration from a JSON file.
        self.config    = util.read_json(p.model_config_path)
        
        # Set a random seed for reproducibility based on the model configuration.
        self.random_state = self.config.get("random_state", 43860714)
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    # Method to predict probability using the trained model.
    def predict_probability(self, input_data: np.array) -> np.array:
        if hasattr(self.model, 'predict_proba'):
            # scikit-learn model
            return self.model.predict_proba(input_data)[:, 1]
        elif hasattr(self.model, 'predict'):
            # TensorFlow/Keras model
            return self.model.predict(input_data).flatten()
        else:
            raise AttributeError("The model does not have a predict method")
        
  	# Method to load a saved model from either a pickle file or an HDF5 file.
    def load_saved_model(self):
        model_path = os.path.join(p.model_path, f"{self.model_name}.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            h5_model_path = os.path.join(p.model_path, f"{self.model_name}.h5")
            self.model = tensorflow.keras.models.load_model(h5_model_path)

    # Method to load SHAP (SHapley Additive exPlanations) values from a pickle file.
    def load_shap_value(self):
        shap_path = os.path.join(p.shap_path, f"{self.model_name}_SHAP_Value.pkl")
        self.shap_values = util.load_model(shap_path)

    # Method to evaluate the model performance on different datasets.
    def model_evaluation(self, backtest: str = "recent"):
        # Load the saved model and preprocess the data.
        self.load_saved_model()
        x_train, y_train, x_test, y_test = self.data_preprocessing()
        benchmark = self.curr_config.get("prediction_benchmark")

        if backtest == "recent":
            # Evaluate model performance on the training set.
            train_predicted_prob = self.predict_probability(x_train)
            # Calculate evaluation metrics on the training set.
            aic_train 			 = aic_calc(y_train, train_predicted_prob, x_train.shape[1])
            bic_train 			 = bic_calc(y_train, train_predicted_prob, x_train.shape[1])
            roc_auc_train 		 = roc_auc_score(y_train, train_predicted_prob)
            train_prediction 	 = np.vectorize(util.map_class)(train_predicted_prob, benchmark)
            res_train, acc_train, precision_train, recall_train, f1_train, cm_train = classification_score(train_prediction,
                                                                                                           y_train,
                                                                                                           roc_auc_train)

            # Generate a report for the training set including evaluation metrics.
            train_report = pd.DataFrame({
                self.model_name: {
                    "Train_Accuracy"	 : acc_train,
                    "Train_Precision"	 : precision_train,
                    "Train_Recall"		 : recall_train,
                    "Train_F1-Score"	 : f1_train,
                    "Train_ROC_AUC_Score": roc_auc_train
                }
            }).fillna(float(0))
            train_report.to_csv(os.path.join(p.model_evaluation_report_raw_path, f"{self.model_name}_train_report.csv"))

            # Evaluate model performance on the testing set.
            test_predicted_prob = self.predict_probability(x_test)
            aic_test 			= aic_calc(y_test, test_predicted_prob, x_test.shape[1])
            bic_test 			= bic_calc(y_test, test_predicted_prob, x_test.shape[1])
            roc_auc_test 		= roc_auc_score(y_test, test_predicted_prob)
            test_prediction 	= np.vectorize(util.map_class)(test_predicted_prob, benchmark)
            res_test, acc_test, precision_test, recall_test, f1_test, cm_test = classification_score(test_prediction,
                                                                                                     y_test,
                                                                                                     roc_auc_test)

            # Generate a report for the testing set including evaluation metrics.
            test_report = pd.DataFrame({
                self.model_name: {
                    "Test_Accuracy": acc_test,
                    "Test_Precision": precision_test,
                    "Test_Recall": recall_test,
                    "Test_F1-Score": f1_test,
                    "Test_ROC_AUC_Score": roc_auc_test
                }
            }).fillna(float(0))
            test_report.to_csv(os.path.join(p.model_evaluation_report_raw_path, f"{self.model_name}_test_report.csv"))

        elif backtest == "stress":
            # Perform evaluation on stress data.
            stress_data_path = os.path.join(p.market_data_output_folder, "stress_period.csv")
            stress_data 	 = pd.read_csv(stress_data_path)

            # Extract features and target variables from stress data.
            x_columns 		 = self.curr_config.get("explanatory_variables")
            x_stress_data 	 = stress_data[x_columns]
            y_column 		 = self.curr_config.get("target_variable")
            y_stress_data 	 = stress_data[y_column]

            # Predict probabilities on stress data.
            stress_predicted_prob = self.predict_probability(x_stress_data)

            # Calculate evaluation metrics on stress data.
            aic_stress 		  = aic_calc(y_stress_data, stress_predicted_prob, len(x_columns))
            bic_stress 		  = bic_calc(y_stress_data, stress_predicted_prob, len(x_columns))
            roc_auc_stress 	  = roc_auc_score(y_stress_data, stress_predicted_prob)
            stress_prediction = np.vectorize(util.map_class)(stress_predicted_prob, benchmark)
            res_stress, acc_stress, precision_stress, recall_stress, f1_stress, cm_stress = classification_score(
                stress_prediction,
                y_stress_data,
                roc_auc_stress)

            # Generate a report for the stress period including evaluation metrics.
            stress_report = pd.DataFrame({
                self.model_name: {
                    "Accuracy"	   : acc_stress,
                    "Precision"	   : precision_stress,
                    "Recall"	   : recall_stress,
                    "F1-Score"	   : f1_stress,
                    "ROC_AUC_Score": roc_auc_stress
                }
            }).fillna(float(0))
            stress_report.to_csv(os.path.join(p.model_evaluation_report_raw_path, f"{self.model_name}_stress_report.csv"))

        else:
            # Handle potential errors or unsupported cases.
            pass

    # Method to backtest a trading strategy using the model's predictions.
    def backtest_strategy(self, transactional_cost: float = 0, backtest : str = "recent"):
        if backtest == "recent":
            # Perform backtesting on recent data.
            ret_colname 		   = util.read_json(p.etl_config_path).get("data_loading").get("EQ_Ticker_Name")
            return_series 		   = self.test_set[ret_colname]
            y_pred_path 		   = os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv")
            y_pred 				   = pd.read_csv(y_pred_path)["final_pred"]
            predicted_returns 	   = y_pred * return_series - y_pred * transactional_cost
            backtest_result 	   = portfolio_evaluation(
                portfolio_returns=pd.Series(predicted_returns),
                benchmark_returns=pd.Series(return_series),
                rf=2.2064041902008153e-05
            )
            backtest_result.columns = [self.model_name]
            backtest_result.to_csv(os.path.join(p.backtest_recent_path, f"{self.model_name}_backtest_recent.csv"))
            
        elif backtest == "stress":
            # Perform backtesting on stress data.
            stress_data_path 		= os.path.join(p.market_data_output_folder, "stress_period.csv")
            stress_data 			= pd.read_csv(stress_data_path)

            # Extract features for stress data.
            x_columns 				= self.curr_config.get("explanatory_variables")
            x_stress_data 			= stress_data[x_columns]

            # Load the model and make predictions.
            self.load_saved_model()
            y_pred_prob 			= self.predict_probability(x_stress_data)
            benchmark 				= self.curr_config.get("prediction_benchmark")
            y_pred_class 			= np.vectorize(util.map_class)(y_pred_prob, benchmark)

            # Evaluate the trading strategy.
            ret_colname 			= util.read_json(p.etl_config_path).get("data_loading").get("EQ_Ticker_Name_Stress")
            return_series 			= stress_data[ret_colname]
            predicted_returns 		= y_pred_class * return_series - y_pred_class * transactional_cost
            backtest_result 	    = portfolio_evaluation(
                portfolio_returns=pd.Series(predicted_returns),
                benchmark_returns=pd.Series(return_series),
                rf=2.2064041902008153e-05
            )
            backtest_result.columns = [self.model_name]
            backtest_result.to_csv(os.path.join(p.backtest_stress_path, f"{self.model_name}_backtest_stress.csv"))
        else:
            # Handle potential errors or unsupported cases.
            pass

    # Method to generate SHAP values for feature importance analysis.
    def generate_shap_value(self):
        # Preprocess data and load the saved model.
        x_train, y_train, x_test, y_test = self.data_preprocessing()
        self.load_saved_model()

        # Initialize SHAP explainer and compute SHAP values.
        explainer 		 			   = shap.Explainer(self.model.predict, x_train)
        self.shap_values 			   = explainer(x_train)
        self.shap_values.feature_names = self.curr_config.get("explanatory_variables")

        # Save SHAP values to a file.
        self.save_shap_value()

    # Method to save the trained model to a pickle file.
    def save_model(self):
        with open(os.path.join(p.model_path, f'{self.model_name}.pkl'), 'wb') as file:
            joblib.dump(self.model, file)

    # Method to save SHAP values to a pickle file.
    def save_shap_value(self):
        with open(os.path.join(p.shap_path, f'{self.model_name}_SHAP_Value.pkl'), 'wb') as file:
            pickle.dump(self.shap_values, file)