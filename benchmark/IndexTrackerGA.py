# Importing necessary libraries
import pandas as pd # Library for data manipulation and analysis
import numpy as np # Library for numerical computations

from PortfolioOptimizer import * # Standalone module for solving quadratic programming problems
from random import randint, sample, choices, uniform # Functions for random sampling and choices
from time import time # Function to track time

# Class definition for IndexTrackerGA, which uses Genetic Algorithm to track an index
class IndexTrackerGA:
    """
    A class to track an index using Genetic Algorithm (GA) optimization.

    Attributes:
        securities_data (pd.DataFrame): Time-series of security returns containing the index.
        max_securities (int):           The maximum number of securities to be retained in the index.
        index_name (str):               The column name of the index series.
        initial_population_size (int):  Initial population size for the GA. Default is 25.
        crossover_probability (float):  Probability of crossover operation in the GA. Default is 1.
        mutation_probability (float):   Probability of mutation operation in the GA. Default is 0.85.
        crossover_cut_point (int):      Location to cut the binary genome of the parent to construct children. Default is 7.
        investment_limits (np.ndarray): Lower and upper bounds of investment in each security. Default is None.
        max_execution_time (float):     Maximum time (in seconds) to allow the algorithm to run. Default is 600 seconds.
        mse_convergence_limit (float):  Convergence criterion for Mean Squared Error (MSE). Default is 5 * (10**(-10)).
    """

    # Constructor method to initialize class attributes
    def __init__(
        self,
        securities_data:       pd.DataFrame,
        max_securities:        int,
        index_name:            str,
        initial_population_size: int = 25,
        crossover_probability: float = 1,
        mutation_probability:  float = 0.85,
        crossover_cut_point:   int = 7,
        investment_limits:     np.ndarray = None,
        max_execution_time:    float = 600,
        mse_convergence_limit: float = 5 * (10**(-10))
    ):
        """
        Initializes the IndexTrackerGA class.

        Args:
            securities_data (pd.DataFrame):           Time-series of security returns containing the index.
            max_securities (int):                     The maximum number of securities to be retained in the index.
            index_name (str):                         The column name of the index series.
            initial_population_size (int, optional):  Initial population size for the GA. Default is 25.
            crossover_probability (float, optional):  Probability of crossover operation in the GA. Default is 1.
            mutation_probability (float, optional):   Probability of mutation operation in the GA. Default is 0.85.
            crossover_cut_point (int, optional):      Location to cut the binary genome of the parent to construct children. Default is 7.
            investment_limits (np.ndarray, optional): Lower and upper bounds of investment in each security. Default is None.
            max_execution_time (float, optional):     Maximum time (in seconds) to allow the algorithm to run. Default is 600 seconds.
            mse_convergence_limit (float, optional):  Convergence criterion for Mean Squared Error (MSE). Default is 5 * (10**(-10)).
        """
        # Assigning input parameters to class attributes
        self.securities_data         = securities_data
        self.max_securities          = max_securities
        self.index_name              = index_name
        self.initial_population_size = initial_population_size
        self.crossover_probability   = crossover_probability
        self.mutation_probability    = mutation_probability
        self.crossover_cut_point     = crossover_cut_point
        self.investment_limits       = investment_limits
        self.max_execution_time      = max_execution_time
        self.mse_convergence_limit   = mse_convergence_limit

    # Method to track the index using Genetic Algorithm optimization
    def track_index(self) -> tuple:
        """
        Method to track the index using Genetic Algorithm optimization.

        Returns:
            tuple: A tuple containing names and weights of selected securities for the index.
        """
        # Record the start time of the algorithm
        start_time 	   = time()
        # Determine the number of securities available for selection
        num_securities = self.securities_data.loc[:, ~self.securities_data.columns.str.match(self.index_name)].shape[1]
        # Generate an initial population of potential solutions (portfolios)
        population     = list(self._generate_initial_population(num_securities, self.initial_population_size, self.max_securities))
        # Initialize variables to track algorithm termination and convergence
        stop           = False
        mse_converged  = False
        results_df     = None

        # Iterate until termination conditions are met
        while not stop:
            children = None
            # Perform crossover operation with a certain probability
            if uniform(0, 1) <= self.crossover_probability:
                # Select parent individuals for crossover
                parents       = choices(population, k=2)
                # Generate children by combining genetic information from parents
                children      = self._perform_crossover(parents, self.crossover_cut_point, self.max_securities)
                # Perform mutation operation with a certain probability
                if uniform(0, 1) <= self.mutation_probability:
                    children  = self._perform_mutation(children)
                # Add children to the population
                population.extend(children)
                # Select top individuals from the population based on objective values
                results_df, mse_converged = self._select_top(population, self.securities_data, self.mse_convergence_limit, self.index_name)
                population                = results_df['population'].to_numpy().tolist()

            # Check if termination conditions are met
            if (time() - start_time) >= self.max_execution_time or mse_converged:
                stop = True

        # Return the names and weights of the selected securities for the index
        return results_df.head(1)['objective_values'][0]

    # Method to generate initial population for the Genetic Algorithm
    def _generate_initial_population(self, num_securities: int, initial_population_size: int, max_securities: int):
        """
        Generate initial population for the Genetic Algorithm.

        Args:
            num_securities (int): Number of securities available for selection.
            initial_population_size (int): Initial population size.
            max_securities (int): Maximum number of securities to be retained in the index.
        """
        # Iterate to create each individual in the initial population
        for _ in range(initial_population_size):
            # Initialize an individual (portfolio) with zeros
            individual       = np.zeros((num_securities,), dtype=int)
            # Randomly select securities to include in the portfolio
            selected_indices = sample(list(range(num_securities)), k=max_securities)
            # Activate the selected securities in the individual
            for activated_index in selected_indices:
                individual[activated_index] = 1
            # Yield the individual for the initial population
            yield individual

    # Method to perform crossover operation in the Genetic Algorithm
    def _perform_crossover(self, parents: list, cut_point: int, max_securities: int):
        """
        Perform crossover operation in the Genetic Algorithm.

        Args:
            parents (list): List of parent individuals.
            cut_point (int): Location to cut the binary genome of the parent.
            max_securities (int): Maximum number of securities to be retained in the index.
        
        Returns:
            list: List of child individuals after crossover operation.
        """
        # Function to correct child individuals to ensure they have the desired number of securities
        def correct_child(child: np.array):
            index_array                = np.array(range(len(child)))

            # Ensure the child does not exceed the maximum number of securities
            while child.sum() > max_securities:
                index_of_change        = index_array[child == 1]
                index_to_change        = sample(index_of_change.tolist(), 1)
                child[index_to_change] = 0

            # Ensure the child includes the required number of securities
            while child.sum() < max_securities:
                index_of_change        = index_array[child == 0]
                index_to_change        = sample(index_of_change.tolist(), 1)
                child[index_to_change] = 1

            return child

        # Function to perform crossover and generate child individuals
        def cut_and_join(parent_indexes: list):
            # Cut the binary genome of parent individuals at the specified cut point
            cut_parent_1 = parents[parent_indexes[0]].tolist()[0:cut_point]
            cut_parent_2 = parents[parent_indexes[1]].tolist()[cut_point:]
            # Combine genetic information from parent individuals to create children
            child        = np.array(cut_parent_1 + cut_parent_2, dtype=int)

            # Correct child individuals to ensure they have the desired number of securities
            if child.sum() != max_securities:
                child = correct_child(child)
                
            return child

        # Generate children by performing crossover operation
        children = []
        children.append(cut_and_join([0, 1]))
        children.append(cut_and_join([1, 0]))

        return children

    # Method to perform mutation operation in the Genetic Algorithm
    def _perform_mutation(self, children: list):
        """
        Perform mutation operation in the Genetic Algorithm.

        Args:
            children (list): List of child individuals.

        Returns:
            list: List of mutated child individuals.
        """
        # Iterate through each child individual and perform mutation
        for child in children:
            index_array 					  = np.array(range(len(child)))

            # Identify activated and deactivated securities in the child individual
            activated_indices 				  = index_array[child == 1]
            deactivated_indices 			  = index_array[child == 0]
            # Randomly select a security to deactivate and activate
            selected_activated_index 		  = sample(activated_indices.tolist(), 1)
            selected_deactivated_index        = sample(deactivated_indices.tolist(), 1)

            # Perform mutation by toggling the activation status of the selected securities
            child[selected_activated_index]   = 0
            child[selected_deactivated_index] = 1
        return children

    # Method to select top individuals from the population based on objective values
    def _select_top(self, population: list, securities_data: pd.DataFrame, mse_limit: float, index_name: str) -> tuple:
        """
        Select top individuals from the population based on objective values.

        Args:
            population (list): List of individuals in the population.
            securities_data (pd.DataFrame): Time-series of security returns containing the index.
            mse_limit (float): Convergence criterion for Mean Squared Error (MSE).
            index_name (str): Column name of the index series.
        
        Returns:
            tuple: A tuple containing selected individuals and whether MSE limit is reached.
        """
        # Create a copy of the securities data
        securities_data_copy = securities_data.copy()

        # Calculate objective values for each individual in the population
        select_data = pd.DataFrame({
            'population': population,
            'objective_values': [self._objective_function(i, securities_data_copy, index_name) for i in population]
        }).sort_values(
            by='objective_values',
            key=lambda col: col.apply(lambda elem: elem['cost_value'])
        ).reset_index()

        # Calculate cost values for each individual
        select_data['cost']          = select_data['objective_values'].apply(lambda x: x['cost_value'])
        # Check if MSE limit is reached for any individual in the population
        select_data['mse_converged'] = select_data['cost'] <= mse_limit
        mse_converged                = any(select_data['mse_converged'][:-2].to_numpy())
        
        # Remove the two worst individuals from the population
        n                            = select_data.shape[0]
        select_data                  = select_data.loc[:, ['population', 'objective_values']].head(n - 2)

        return select_data, mse_converged

    # Method to compute objective function value for an individual
    def _objective_function(self, individual: np.ndarray, securities_data: pd.DataFrame, index_name: str) -> tuple:
        """
        Compute objective function value for an individual.

        Args:
            individual (np.ndarray): Individual representing selection of securities.
            securities_data (pd.DataFrame): Time-series of security returns containing the index.
            index_name (str): Column name of the index series.
        
        Returns:
            tuple: A tuple containing weights, names, and cost value of selected securities.
        """
        # Create a copy of the securities data
        securities_data_copy      = securities_data.copy()
        # Select data corresponding to the selected securities in the individual
        selected_data 			  = securities_data_copy.loc[:, ~securities_data_copy.columns.str.match(index_name)]
        selected_data 			  = selected_data.loc[:, individual == 1]
        selected_data[index_name] = securities_data[index_name]

        # Solve quadratic programming problem to determine weights of selected securities
        Port_Opt 				  = PortfolioOptimizer(returns_data=selected_data, index_column=index_name)
        solution 				  = Port_Opt.optimize_portfolio()

        # Return weights, names, and cost value of selected securities
        return {'weights':    solution['x'],
                'names':      Port_Opt.assets,
                'cost_value': solution['cost value']}