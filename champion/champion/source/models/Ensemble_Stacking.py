import pandas as pd
import numpy as np
import os

from champion.source.models._mainModeling import _mainModeling
from sklearn.ensemble import StackingClassifier
from champion.source.util import _util as util
from champion.config import _paths as p

class Ensemble_Stacking(_mainModeling):
    """A class for training and evaluating a Stacking Classifier ensemble model."""

    def __init__(self):
        """Initialize the Ensemble_Stacking class."""
        super().__init__()
        self.curr_config = self.config.get("Ensemble_Stacking")
        self.model_name  = self.curr_config.get("name")

    def data_preprocessing(self):
        """Preprocess the data."""
        x_column = self.curr_config.get("explanatory_variables")
        y_column = self.curr_config.get("target_variable")

        # convert target variable to categorical
        y_train = pd.Categorical(self.train_set[y_column])
        y_test  = pd.Categorical(self.test_set[y_column])

        x_train, y_train = np.array(self.train_set[x_column]), np.array(y_train)
        x_test, y_test   = np.array(self.test_set[x_column]), np.array(y_test)

        x_train, y_train = x_train.reshape(-1, len(x_column)), y_train
        x_test, y_test   = x_test.reshape(-1, len(x_column)), y_test

        return x_train, y_train, x_test, y_test

    def get_estimators(self) -> list:
        """Get base estimators for the Stacking Classifier."""
        all_estimators = []
        estimators 	   = self.curr_config.get("estimator")
        model_path 	   = p.model_path
        for est in estimators:
            model_name 		= self.config.get(est).get("name")
            model_full_path = os.path.join(model_path, model_name + '.pkl')
            model 			= util.load_model(model_full_path)
            all_estimators.append((model_name, model))
        return all_estimators

    def model_training(self):
        """Train the Stacking Classifier ensemble model."""
        x_train, y_train, x_test, y_test = self.data_preprocessing()
        all_estimators 					 = self.get_estimators()

        self.model 						 = StackingClassifier(estimators=all_estimators, stack_method="predict_proba")
        self.model.fit(x_train, y_train)

        y_pred 							 = self.model.predict_proba(x_test)

        y_pred_df 				= pd.DataFrame(y_pred, columns=["Class 0", "Class 1"], index=self.test_set["Date"])
        benchmark 				= self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["Class 1"].apply(lambda x: int(x >= benchmark))

        # output the prediction
        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))