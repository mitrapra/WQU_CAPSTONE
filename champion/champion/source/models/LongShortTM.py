import tensorflow

import pandas as pd
import numpy as np
import statistics
import time
import os

import tensorflow.keras.initializers
import tensorflow.keras

from sklearn import metrics

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from bayes_opt import BayesianOptimization

from champion.source.models._mainModeling import _mainModeling
from champion.source.models._metricsNN import _metricsNN
from champion.config import _paths as p

import warnings
warnings.filterwarnings("ignore")

class LongShortTM(_mainModeling, _metricsNN):
    
    def __init__(self):
        super().__init__()
        self.curr_config = self.config.get("LSTM")
        self.model_name  = self.curr_config.get("name")
        tensorflow.random.set_seed(self.random_state)
        
        self.EPOCHS   = 50
        self.PATIENCE = 10
        
    def data_preprocessing(self):
        x_column = self.curr_config.get("explanatory_variables")
        y_column = self.curr_config.get("target_variable")
        
        X_train, y_train = np.array(self.train_set[x_column]), np.array(self.train_set[y_column])
        X_test, y_test   = np.array(self.test_set[x_column]), np.array(self.test_set[y_column])
                                                                       
        X_train, y_train = X_train.reshape(-1, len(x_column)), y_train.reshape((-1, 1))
        X_test, y_test   = X_test.reshape(-1, len(x_column)), y_test.reshape((-1, 1))
        
        return X_train, y_train, X_test, y_test

    def generate_model(self, dropout, neuronPct, neuronShrink):
        X_train, y_train, X_test, y_test = self.data_preprocessing()
        
        neuronCount                      = int(neuronPct * 1000)
        model                            = Sequential()
        layer                            = 0
        while neuronCount > 25 and layer < 100:
            if layer == 0:
                model.add(LSTM(neuronCount,
                               input_shape=(X_train.shape[1], 1),
                               activation='sigmoid'))
            else:
                model.add(Dense(neuronCount, 
                                activation='sigmoid')) 
            layer += 1
            
            model.add(Dropout(dropout))
            neuronCount = int(neuronCount * neuronShrink)
            
        model.add(Dense(y_train.shape[1], activation='softmax'))
        
        return model

    def evaluate_network(self, dropout, learning_rate, neuronPct, neuronShrink):
        mean_benchmark = []
        epochs_needed  = []
        num = 0
        
        start_time = time.time()
        num       += 1
        X_train, y_train, X_test, y_test = self.data_preprocessing()

        model = self.generate_model(dropout,
                                    neuronPct,
                                    neuronShrink)
        model.compile(loss     ='binary_crossentropy',
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics  =['accuracy'])

        monitor = EarlyStopping(monitor             ='val_loss',
                                min_delta           =1e-3, 
                                patience            =self.PATIENCE, 
                                verbose             =0, 
                                mode                ='max', 
                                restore_best_weights=True)
        model.fit(X_train, 
                  y_train, 
                  validation_data=(X_test, y_test), 
                  callbacks      =[monitor], 
                  verbose        =0, 
                  epochs         =self.EPOCHS)

        epochs = monitor.stopped_epoch
        epochs_needed.append(epochs)

        y_pred       = model.predict(X_test).reshape(-1)
        col_mean     = np.nanmean(y_pred, axis = 0)
        indx         = np.where(np.isnan(y_pred))
        
        if indx[0].size == y_pred.shape[0]:
            y_pred          = np.full(y_pred.shape[0], 0.85)
        elif indx[0].size < y_pred.shape[0]:
            y_pred[indx[0]] = np.take(col_mean, indx[0])
        else:
            y_pred          = y_pred
            
        score     = metrics.log_loss(y_test, y_pred)
        mean_benchmark.append(score)
        m1        = statistics.mean(mean_benchmark)
        m2        = statistics.mean(epochs_needed)
        mdev      = statistics.pstdev(mean_benchmark)
        time_took = time.time() - start_time

        tensorflow.keras.backend.clear_session()
        return -m1

    def model_training(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pbounds = {'dropout'      : (0.0, 0.499),
                   'learning_rate': (0.0, 0.1),
                   'neuronPct'    : (0.01, 1),
                   'neuronShrink' : (0.01, 1)}
        optimizer = BayesianOptimization(f=self.evaluate_network, 
                                         pbounds=pbounds, 
                                         verbose=0, 
                                         random_state=self.random_state)
        
        optimizer.maximize(init_points=45, n_iter=10)
        
        best_params  = optimizer.max['params']

        self.model   = self.generate_model(dropout     =best_params['dropout'],
                                           neuronPct   =best_params['neuronPct'],
                                           neuronShrink=best_params['neuronShrink'])
        self.model.compile(loss     ='binary_crossentropy',
                           optimizer=Adam(learning_rate=best_params['learning_rate']),
                           metrics  =['accuracy'])
        
        X_train, y_train, X_test, y_test = self.data_preprocessing()
        self.model.fit(X_train, y_train, epochs=self.EPOCHS, verbose=0)
        
        y_pred_final = self.model.predict(X_test).reshape(-1)
        y_pred_df    = pd.DataFrame({
            "Class 0": 1 - y_pred_final,
            "Class 1": y_pred_final
        }, index=self.test_set["Date"])

        benchmark = self.curr_config.get("prediction_benchmark")
        y_pred_df["final_pred"] = y_pred_df["Class 1"].apply(lambda x: int(x >= benchmark))

        y_pred_df.to_csv(os.path.join(p.model_prediction_path, f"{self.model_name}_pred.csv"))