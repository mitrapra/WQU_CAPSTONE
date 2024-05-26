# Importing necessary libraries
import cvxopt
from cvxopt import solvers, matrix

import pandas as pd
import numpy as np

# Defining a class to optimize a portfolio based on given returns data
class PortfolioOptimizer:
    # Constructor to initialize the optimizer with returns data, constraints, and index column name
    def __init__(self, returns_data: pd.DataFrame, constraints: np.ndarray=None, index_column: str='index'):
        self.returns_data = returns_data.copy()  # Copying the returns data into the optimizer
        self.index_column = index_column  		 # Name of the column used as index in the returns data
        
        # List of asset names in the returns data
        self.assets 	  = returns_data.loc[:, ~returns_data.columns.str.match(self.index_column)].columns.to_numpy().tolist()
        self.constraints  = constraints  		 # Constraints for optimization (optional)
    
    # Method to compute the covariance matrix of the returns data
    def _covariance_matrix(self):
        returns_data = self.returns_data.copy()  		# Copying the returns data
        returns_data = returns_data.loc[:, ~returns_data.columns.str.match(self.index_column)]  # Excluding the index column
        num_assets   = returns_data.shape[1]  			# Number of assets
        num_periods  = returns_data.shape[0]  			# Number of periods
        asset_names  = returns_data.columns.to_numpy()  # Names of the assets
        
        cov_matrix = np.zeros((num_assets, num_assets))  # Initializing the covariance matrix object
        
        # Computing covariance between each pair of assets
        for i, asset_i in enumerate(asset_names):
            for j, asset_j in enumerate(asset_names):
                value = np.dot(returns_data[asset_i].copy().to_numpy(),  			   # Dot product of returns
                               returns_data[asset_j].copy().to_numpy()) / num_periods  # Normalizing by number of periods
                cov_matrix[i, j] = value  # Setting covariance value at the (i,j) position
                
        return cov_matrix  # Returning the covariance matrix
    
    # Method to compute expected returns of each asset
    def _expected_returns(self):
        returns_data 	 = self.returns_data.copy() # Copying the returns data
        num_assets 	 	 = returns_data.loc[:, ~returns_data.columns.str.match(self.index_column)].shape[1] # Number of assets
        num_periods  	 = returns_data.shape[0] # Number of periods
        asset_names  	 = returns_data.loc[:, ~returns_data.columns.str.match(self.index_column)].columns.to_numpy() # Names of the assets
        
        expected_returns = np.zeros(num_assets)  # Initializing expected returns
        # Computing expected return for each asset
        for i, asset in enumerate(asset_names):
            value = np.dot(returns_data[asset].copy().to_numpy(), # Dot product of returns
                           returns_data[self.index_column].copy().to_numpy()) / num_periods # Normalizing by number of periods
            expected_returns[i] = value # Setting expected return value
            
        return expected_returns # Returning the expected returns
    
    # Method to compute equality constraints for optimization
    def _equality_constraints(self):
        returns_data = self.returns_data.copy() # Copying the returns data
        num_assets 	 = returns_data.loc[:, ~returns_data.columns.str.match(self.index_column)].shape[1] # Number of assets
        A 			 = np.repeat(1, num_assets) # Coefficients for equality constraint
        b 			 = np.array([1]) # Right-hand side value for equality constraint
        
        A 			 = np.reshape(A, (1, num_assets)) # Reshaping A
        b 			 = np.reshape(b, (1, 1)) # Reshaping b
        
        return A, b # Returning equality constraints
    
    # Method to compute inequality constraints for optimization
    def _inequality_constraints(self):
        returns_data = self.returns_data.copy() # Copying the returns data
        num_assets 	 = returns_data.loc[:, ~returns_data.columns.str.match(self.index_column)].shape[1] # Number of assets
        G 			 = -np.identity(num_assets) # Coefficients for inequality constraint
        h 			 = np.repeat([0], num_assets).transpose() # Right-hand side values for inequality constraint
        h 			 = np.reshape(h, (num_assets, 1)) # Reshaping h
        
        return G, h # Returning inequality constraints
    
    # Method to optimize the portfolio
    def optimize_portfolio(self):
        returns_data 	  = self.returns_data.copy() # Copying the returns data
        num_assets 		  = returns_data.loc[:, ~returns_data.columns.str.match(self.index_column)].shape[1] # Number of assets
        
        covariance_matrix = matrix(self._covariance_matrix(), tc='d') # Computing covariance matrix
        expected_returns  = matrix(self._expected_returns(), tc='d') # Computing expected returns
        
        A, b 			  = self._equality_constraints() # Computing equality constraints
        A 				  = matrix(A, tc='d') # Converting A to matrix
        b 				  = matrix(b, tc='d') # Converting b to matrix
        
        G, h 			  = self._inequality_constraints() # Computing inequality constraints
        G 				  = matrix(G, tc='d')  # Converting Z to matrix
        h 				  = matrix(h, tc='d')  # Converting h to matrix
        
        # Solving the optimization problem using quadratic programming
        optimized_portfolio = solvers.qp(P=covariance_matrix, 
                                         q=expected_returns, # Objective
                                         G=G, h=h, # Linear inequalities
                                         A=A, b=b) # Linear constraints
        
        optimized_portfolio['x'] = np.array(optimized_portfolio['x'], dtype=float) # Extracting optimized weights
        
        # Calculating cost value (e.g., standard deviation) of the optimized portfolio
        stock_data 				 = returns_data.loc[:, ~returns_data.columns.str.match(self.index_column)].to_numpy() # Stock data
        weights 				 = optimized_portfolio['x'] # Optimized weights
        portfolio_results 		 = np.matmul(stock_data, weights).flatten() # Portfolio returns
        cost_value 				 = np.sqrt(np.mean((returns_data[self.index_column].to_numpy() - portfolio_results)**2)) # Standard deviation
        
        optimized_portfolio['cost value'] = cost_value # Adding cost value to the optimized portfolio
        
        return optimized_portfolio # Returning the optimized portfolio