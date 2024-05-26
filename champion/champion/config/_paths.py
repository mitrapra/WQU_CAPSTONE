import os

# Define the current working directory
cwd = r"champion"

# For ETL (Extract, Transform, Load) processes:

# Path to the ETL configuration file
etl_config_path = os.path.join(cwd, "config", "_etl.json")

# Path to the output folder for processed data
output_folder = os.path.join(cwd, "data")

# Path to the folder for storing market data
market_data_output_folder = os.path.join(output_folder, "market_data")

# For Modeling processes:

# Path to the model configuration file
model_config_path = os.path.join(cwd, "config", "_models.json")

# Paths to the training and testing datasets
train_set_path = os.path.join(market_data_output_folder, "train_set.csv")
test_set_path = os.path.join(market_data_output_folder, "test_set.csv")

# Path to the folder for storing model predictions
model_prediction_path = os.path.join(output_folder, "_predictions")

# Path to the folder for storing trained models
model_path = os.path.join(output_folder, "_models")

# For Model Evaluation:

# Path to the folder for storing model evaluation reports
model_evaluation_report_path = os.path.join(output_folder, "_evaluation")

# Path to the folder for storing raw model evaluation reports
model_evaluation_report_raw_path = os.path.join(model_evaluation_report_path, "_raw")

# Path to the folder for storing compiled model evaluation reports
model_evaluation_report_compiled_path = os.path.join(model_evaluation_report_path, "_compiled")

# Path to the folder for storing recent backtest results
backtest_recent_path = os.path.join(model_evaluation_report_path, "_backtest", "_recent")

# Path to the folder for storing stress test results
backtest_stress_path = os.path.join(model_evaluation_report_path, "_backtest", "_stress")

# Path to the folder for storing SHAP (SHapley Additive exPlanations) values
shap_path = os.path.join(model_evaluation_report_path, "_shap")