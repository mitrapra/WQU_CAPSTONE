import empyrial
import numpy as np

from empyrial import empyrial, Engine

class PortfolioOptimizationRiskParameterTuning:
    """
    This class is designed to optimize investment portfolios by experimenting with various risk management strategies
    and rebalancing frequencies. It leverages the empyrial library to evaluate portfolio performance based on different
    configurations, aiming to find the optimal setup that maximizes returns while managing risk effectively.
    """

    def __init__(self, risk_managers, rebalancing_periods=None, end_date: str="2024-03-31", start_date: str="2000-01-04", LTCM: bool=True):
        """
        Initializes the PortfolioOptimizationRiskParameterTuning class with specified risk managers, rebalancing periods,
        start date, and an end date for the optimization process.

        Args:
            risk_managers (list): A list of dictionaries, each specifying a risk manager and its parameters to be tested.
            rebalancing_periods (list, optional): A list of strings representing the rebalancing frequencies to be tested.
                                                   Defaults to ["monthly", "quarterly", "6m"] if not provided.
            end_date (str, optional): The end date for the portfolio evaluation period in YYYY-MM-DD format. Defaults to "2024-03-31".
            start_date (str, optional): The start date for the portfolio evaluation period in YYYY-MM-DD format. Defaults to "2000-01-04".
            LTCM (bool, optional): A boolean flag to adjust the start date for the evaluation period based on a condition.
                                   If set to False, the start date is overridden to "2022-01-01".

        This constructor initializes the class with the necessary parameters for portfolio optimization, including the
        risk management strategies to be tested, the rebalancing periods, and the evaluation period defined by the start
        and end dates. The LTCM flag allows for conditional adjustment of the start date, providing flexibility in setting
        the evaluation period based on specific criteria.
        """
        self.risk_managers = risk_managers  # Stores the risk management strategies to be tested
        self.end_date      = end_date       # The final date for evaluating portfolio performance
        self.start_date    = start_date if LTCM else "2022-01-01" # The starting date for evaluating portfolio performance, adjusted based on the LTCM flag
        
        # Sets the rebalancing periods for the portfolio, with a default set if none is provided
        self.rebalancing_periods = rebalancing_periods if rebalancing_periods else ["monthly", "quarterly", "6m", "1y"]

    def optimize_portfolio(self, _listStocks: list, risk_free_rate: float):
        """
        Optimizes portfolios by iterating over combinations of risk managers and rebalancing periods, evaluating
        each configuration's performance to identify the best setup.

        Args:
            _listStocks (list): A list of stock tickers to be included in the portfolio.
            risk_free_rate (float): The risk-free rate to be used in the portfolio performance evaluation, typically
                                    the yield on 3-month US Treasury bills.

        Returns:
            tuple: Returns a tuple of lists containing the configurations tested (risk managers and rebalancing periods)
                   and their corresponding performance metrics (ALPHA, STABILITY, CAGR).

        This method systematically explores different configurations of risk management strategies and rebalancing frequencies
        to optimize the portfolio. It evaluates the performance of each configuration against the risk-free rate and records
        the performance metrics, facilitating the identification of the most effective setup for maximizing returns while
        managing risk.
        """
        rMngr              = []  # List to store the risk manager configurations tested
        rebalance_periods  = []  # List to store the rebalancing periods tested
        ALPHA_values       = []  # List to store the ALPHA values obtained
        STABILITYF_values  = []  # List to store the STABILITY values obtained
        CAGRF_values       = []  # List to store the CAGR values obtained

        # Iterate through each risk manager and rebalancing period combination
        for risk_manager_index in range(len(self.risk_managers)):
            for rebalance_period in self.rebalancing_periods:
                # Initialize the portfolio with the current configuration
                portfolio = self._initialize_portfolio(_listStocks, risk_manager_index, rebalance_period)

                # Evaluate the portfolio's performance with the current configuration
                ALPHA, STABILITY, CAGR = self._evaluate_portfolio_performance(portfolio, risk_free_rate)

                # Store the results for analysis
                rMngr.append(self.risk_managers[risk_manager_index])  # Append the risk manager configuration used
                rebalance_periods.append(rebalance_period)  # Append the rebalancing period used
                ALPHA_values.append(ALPHA)  # Append the ALPHA value obtained
                STABILITYF_values.append(STABILITY)  # Append the STABILITY value obtained
                CAGRF_values.append(CAGR)  # Append the CAGR value obtained
                
        STABILITY_values = []
        CAGR_values		 = []
        
        for i in np.arange(len(STABILITYF_values)):
            S = float(STABILITYF_values[i])
            C = round(float(CAGRF_values[i].replace('%', ''))/100,4)
            STABILITY_values.append(S)
            CAGR_values.append(C)
            
        return rMngr, rebalance_periods, ALPHA_values, STABILITY_values, CAGR_values

    def _initialize_portfolio(self, _listStocks, risk_manager_index, rebalance_period):
        """
        Initializes a portfolio with a given set of parameters, preparing it for performance evaluation.

        Args:
            _listStocks (list): The list of stock tickers to be included in the portfolio.
            risk_manager_index (int): The index of the risk manager in the list of risk managers to be applied.
            rebalance_period (str): The rebalancing frequency for the portfolio.

        Returns:
            dict: A dictionary of portfolio parameters ready to be passed to the portfolio evaluation engine.

        This method sets up the portfolio based on the specified stocks, risk manager, and rebalancing frequency. It
        prepares a dictionary of parameters that define the portfolio's configuration, which is then used to evaluate
        the portfolio's performance. This setup process is crucial for ensuring that each portfolio configuration is
        consistently evaluated under the same conditions.
        """
        # Define the portfolio parameters based on the inputs
        portfolio_params = {
            "start_date"  : self.start_date,  # The start date for the portfolio evaluation period
            "end_date"    : self.end_date,  # The end date for the portfolio evaluation period
            "portfolio"   : _listStocks,  # The list of stocks to be included in the portfolio
            "optimizer"   : "EF",  # The optimizer to be used, here "EF" stands for the Global Efficient Frontier
            "rebalance"   : rebalance_period,  # The rebalancing frequency for the portfolio
            "benchmark"   : ["SPY"],  # The benchmark against which to compare the portfolio, here "SPY" for S&P 500
            "max_weights" : 0.35,  # The maximum weight allowed for any single asset in the portfolio
            "min_weights" : -0.05,  # The minimum weight allowed for any single asset in the portfolio
            "risk_manager": self.risk_managers[risk_manager_index]  # The risk manager configuration to be applied
        }

        return portfolio_params

    def _evaluate_portfolio_performance(self, portfolio_params, risk_free_rate):
        """
        Evaluates the performance of a portfolio based on the provided parameters and risk-free rate.

        Args:
            portfolio_params (dict): The parameters defining the portfolio to be evaluated.
            risk_free_rate (float): The risk-free rate to be used in the evaluation.

        Returns:
            tuple: Returns a tuple containing the ALPHA, STABILITY, and CAGR values for the evaluated portfolio.

        This method uses the empyrial library to evaluate the performance of the configured portfolio against the specified
        risk-free rate. It extracts key performance metrics, including ALPHA, STABILITY, and CAGR, which are essential for
        assessing the effectiveness of the portfolio's risk management and rebalancing strategy. This evaluation process
        is critical for identifying the optimal portfolio configuration.
        """
        # Initialize the portfolio with the given parameters
        expEF_portfolio = Engine(**portfolio_params)
        # Evaluate the portfolio's performance using the empyrial function
        empyrial(
            expEF_portfolio,
            rf=risk_free_rate,  # Set the risk-free rate for the evaluation
            confidence_value=0.99  # Set the confidence level for the evaluation
        )
        # Extract the performance metrics from the evaluation
        ALPHA     = empyrial.AL         # ALPHA value indicating the portfolio's excess return on a risk-adjusted basis
        STABILITY = empyrial.STABILITY  # STABILITY value indicating the consistency of the portfolio's returns
        CAGR      = empyrial.CAGR       # CAGR (Compound Annual Growth Rate) indicating the mean annual growth rate of the portfolio

        return ALPHA, STABILITY, CAGR