# Import necessary modules
import json  # Module for working with JSON data
import pickle  # Module for serializing and deserializing Python objects
import os  # Module for interacting with the operating system

# Import configuration properties from a custom module
from champion.config import _paths as p

# Function to read JSON files and return the content as a dictionary
def read_json(json_file_path: str) -> dict:
    with open(json_file_path) as f:  # Open the JSON file
        f = json.load(f)  # Load JSON content into a dictionary
    return f  # Return the dictionary

# Function to load a machine learning model from a pickle file
def load_model(pkl_file_path: str):
    return pickle.load(open(pkl_file_path, "rb"))  # Load and return the pickled model

# Function to map a class probability to a binary class based on a benchmark
def map_class(class_prob: float, benchmark: float) -> int:
    return int(class_prob >= benchmark)  # Return 1 if class probability is greater than or equal to benchmark, otherwise return 0

# Function to create a directory if it does not exist
def make_dir(directory: str):
    if not os.path.exists(directory):  # Check if the directory does not exist
        os.makedirs(directory)  # Create the directory and any necessary parent directories

# Function to manage directories as per configuration properties
def dir_management():
    # Create necessary directories for data processing and model evaluation
    make_dir(p.output_folder)  # Create the output directory
    make_dir(p.market_data_output_folder)  # Create the market data output directory
    make_dir(p.model_prediction_path)  # Create the model prediction directory
    make_dir(p.model_evaluation_report_path)  # Create the model evaluation report directory
    make_dir(p.model_evaluation_report_raw_path)  # Create the raw model evaluation report directory
    make_dir(p.model_evaluation_report_compiled_path)  # Create the compiled model evaluation report directory
    make_dir(p.backtest_recent_path)  # Create the recent backtest directory
    make_dir(p.backtest_stress_path)  # Create the stress test directory
    make_dir(p.shap_path)  # Create the SHAP directory
    make_dir(p.model_path)  # Create the folder to store the trained models