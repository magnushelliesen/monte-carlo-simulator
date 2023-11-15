import pandas as pd
import numpy as np
from numpy.linalg import eig, inv
from arch import arch_model

# First iteration of class
class MonteCarlo():
    """
    A Monte Carlo simulation class. More TBA

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing time series data.

    Methods:
    --------
    fit():
        Perform the Monte Carlo simulation to generate orthogonal disturbances with GARCH models.

    Returns:
    --------
    MonteCarloResult
        An object containing the results of the Monte Carlo simulation.

    See Also:
    ---------
    MonteCarloResult : The resulting object containing the simulation results.
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._df_mean = df.mean(axis=0)

    @property
    def df(self):
        return self._df

    def fit(self) -> None:
        # Deep copying DataFrame and calculating covariance matrix
        orthog_disturbances_df = self.df-self._df_mean
        array = orthog_disturbances_df.to_numpy()
        covariance_matrix = np.cov(array.T)

        # Calculating eigenvalues and vectors from covariance matrix
        eigen_values, eigen_vectors = eig(covariance_matrix)
        eigen_values = np.diag(eigen_values)
        
        # Calculating weights and orthogonal disturbances with unit variance
        combination_matrix = (eigen_values**0.5).dot(eigen_vectors.T)
        array[:, :] = array.dot(inv(combination_matrix))

        # Renaming columns
        orthog_disturbances_df.columns = [i for i in range(len(orthog_disturbances_df.columns))]

        # Setting up GARCH-model for each orthogonal disturbance
        garch_models = {i: arch_model(orthog_disturbances_df[i], vol='garch', p=1, o=0, q=1, rescale=False)
                  for i in orthog_disturbances_df.columns}

        # Calculating model fits
        garch_fits = {i: model.fit(disp='off') for i, model in garch_models.items()}

        # Calcutating conditional volatility
        conditional_volatility_df = pd.DataFrame()
        for i in range(len(orthog_disturbances_df.columns)):
            conditional_volatility_df[i] = garch_fits.get(i).conditional_volatility

        # calculating orthogonal disturbances normalized by GARCH standard deviation
        norm_orthog_disturbances_df = orthog_disturbances_df/conditional_volatility_df

        # Returning MonteCarloResult object with all necessary stuff
        return (
            MonteCarloResult(
                combination_matrix,
                orthog_disturbances_df,
                norm_orthog_disturbances_df,
                conditional_volatility_df,
                garch_models,
                garch_fits,
                self._df_mean,
                self.df.columns
            )
        )


class MonteCarloResult():
    """
    A class to encapsulate the results of a Monte Carlo simulation using GARCH models.

    Parameters:
    -----------
    combination_matrix : numpy.ndarray
        Matrix representing the combination of eigenvalues and eigenvectors.
        
    orthog_disturbances_df : pandas.DataFrame
        DataFrame containing orthogonal disturbances.

    norm_orthog_disturbances_df : pandas.DataFrame
        DataFrame containing normalized orthogonal disturbances.

    conditional_volatility_df : pandas.DataFrame
        DataFrame containing conditional volatility estimates.

    garch_models : dict
        A dictionary of GARCH models used for each orthogonal disturbance.

    garch_fits : dict
        A dictionary of GARCH model fits for each orthogonal disturbance.

    Attributes:
    -----------
    combination_matrix : numpy.ndarray
        Matrix representing the combination of eigenvalues and eigenvectors.

    orthog_disturbances_df : pandas.DataFrame
        DataFrame containing orthogonal disturbances.

    norm_orthog_disturbances_df : pandas.DataFrame
        DataFrame containing normalized orthogonal disturbances.

    conditional_volatility_df : pandas.DataFrame
        DataFrame containing conditional volatility estimates.

    garch_models : dict
        A dictionary of GARCH models used for each orthogonal disturbance.

    garch_fits : dict
        A dictionary of GARCH model fits for each orthogonal disturbance.

    Methods:
    --------
    forecast(n_periods: int) -> pd.DataFrame:
        Forecast conditional volatility for a specified number of periods ahead.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing conditional volatility forecasts.

    See Also:
    ---------
    MonteCarlo : The class for performing the Monte Carlo simulation.
    """
    def __init__(
        self,
        combination_matrix,
        orthog_disturbances_df,
        norm_orthog_disturbances_df,
        conditional_volatility_df,
        garch_models,
        garch_fits,
        mean_df,
        colnames
        ):
        self._combination_matrix = combination_matrix
        self._orthog_disturbances_df = orthog_disturbances_df
        self._norm_orthog_disturbances_df = norm_orthog_disturbances_df
        self._conditional_volatility_df = conditional_volatility_df
        self._garch_models = garch_models
        self._garch_fits = garch_fits
        self._mean_df = mean_df
        self._colnames = colnames

    @property
    def combination_matrix(self):
        return self._combination_matrix

    @property
    def orthog_disturbances_df(self):
        return self._orthog_disturbances_df

    @property
    def norm_orthog_disturbances_df(self):
        return self._norm_orthog_disturbances_df

    @property
    def conditional_volatility_df(self):
        return self._conditional_volatility_df

    def forecast(self, n_periods: int, n_simulations) -> pd.DataFrame:
        """
        UNDER CONSTRUCION
        Docstring will come
        """

        conditional_volatility_forecast_df = pd.DataFrame()

        # Make GARCH forecast of volatility
        for i, fit in self._garch_fits.items():
            conditional_volatility_forecast_df[i] = (
                fit
                .forecast(horizon=n_periods, reindex=False)
                .variance
                .T
            )

        # Set index = 0, 1, 2, ...
        conditional_volatility_forecast_df.index = range(n_periods)

        # Calculate simulations
        simulations = {}

        # Draw from normalized orthogonal disturbances
        for i in range(n_simulations):
            orthog_disturbance_draws = pd.DataFrame()
            for j, col in enumerate(self.norm_orthog_disturbances_df.columns):
                norm_orthog_disturbance_draw = (
                    pd.Series(
                        np.random.choice(
                            self.norm_orthog_disturbances_df[col],
                            size=n_periods,
                            replace=True
                            ),
                        index=range(n_periods)
                    )
                )

                # Introduce forecast of conditional volatility
                orthog_disturbance_draw = (
                    norm_orthog_disturbance_draw
                    *conditional_volatility_forecast_df[j]
                )

                # Store in DataFrame
                orthog_disturbance_draws[col] = orthog_disturbance_draw

            # Weigh together orthogonal disturbances to get origianl
            simulations[i] = orthog_disturbance_draws.dot(self.combination_matrix)
            simulations.get(i).columns = self._colnames
            simulations[i] = simulations[i]+self._mean_df

        return simulations
