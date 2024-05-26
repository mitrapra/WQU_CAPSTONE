import pandas as pd
import numpy as np
import os

from sklearn import linear_model

from champion.source.model_evaluation._evaluation_metrics import classification_score
from champion.source.models._mainModeling import _mainModeling
from champion.source.util import _util as util
from champion.config import _paths as p

class Logistic_Regression(_mainModeling):
    """A class for training and evaluating a logistic regression model."""

    def __init__(self):
        """Initialize the Logistic_Regression class."""
        super().__init__()
        self.curr_config = self.config.get("Logistic_Regression")
        self.model_name  = self.curr_config.get("name")

    def data_preprocessing(self):
        """Preprocess the data."""
        x_column = self.curr_config.get("explanatory_variables")
        y_column = self.curr_config.get("target_variable")

        # Extract the explanatory and target variables from the train and test sets
        x_train, y_train = np.array(self.train_set[x_column]), np.array(self.train_set[y_column])
        x_test, y_test 	 = np.array(self.test_set[x_column]), np.array(self.test_set[y_column])

        # Reshape the data arrays if necessary
        x_train, y_train = x_train.reshape(-1, len(x_column)), y_train
        x_test, y_test 	 = x_test.reshape(-1, len(x_column)), y_test

        return x_train, y_train, x_test, y_test

    def model_training(self):
        """Train the logistic regression model."""
        x_train, y_train, x_test, y_test = self.data_preprocessing()

        # Train the logistic regression model
        self.model = linear_model.LogisticRegression()
        self.model.fit(x_train, y_train)

        # Predict the probabilities
        y_pred 	   = self.model.predict_proba(x_test)

        # Generate the predicted class based on the benchmark
        y_pred_df  = pd.DataFrame(y_pred, columns=["Class 0", "Class 1"], index=self.test_set["Date"])
        benchmark  = self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["Class 1"].apply(lambda x: int(x >= benchmark))

        # Output the prediction
        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))