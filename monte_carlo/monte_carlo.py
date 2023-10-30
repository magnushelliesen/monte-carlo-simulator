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

    @property
    def df(self):
        return self._df

    def fit(self):
        # Deep copying DataFrame and calculating covariance matrix
        orthog_disturbances_df = self.df.copy(deep=True)
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
            conditional_volatility_df[i] = garch_fits[i].conditional_volatility

        # calculating orthogonal disturbances normalized by GARCH standard deviation
        norm_orthog_disturbances_df = (
            (orthog_disturbances_df-orthog_disturbances_df.mean(axis=0))
            /conditional_volatility_df
            +orthog_disturbances_df.mean(axis=0)
        )

        # Returning MonteCarloResult object with all necessary stuff
        return (
            MonteCarloResult(
                combination_matrix,
                orthog_disturbances_df,
                norm_orthog_disturbances_df,
                conditional_volatility_df,
                garch_models,
                garch_fits
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
        garch_fits
        ):
        self._combination_matrix = combination_matrix
        self._orthog_disturbances_df = orthog_disturbances_df
        self._norm_orthog_disturbances_df = norm_orthog_disturbances_df
        self._conditional_volatility_df = conditional_volatility_df
        self._garch_models = garch_models
        self._garch_fits = garch_fits

    @property
    def scaling_matrix(self):
        return self._scaling_matrix

    @property
    def orthog_disturbances_df(self):
        return self._orthog_disturbances_df

    @property
    def norm_orthog_disturbances_df(self):
        return self._norm_orthog_disturbances_df

    @property
    def conditional_volatility_df(self):
        return self._conditional_volatility_df

    def forecast(self, n_periods: int) -> pd.DataFrame:
        """
        UNDER CONSTRUCTION
        """
        # Forecast conditional volatility n_periods ahead in time
        conditional_volatility_forecast_df = pd.DataFrame()
        for i, fit in self._garch_fits.items():
            conditional_volatility_forecast_df[i] = (
                fit
                .forecast(horizon=n_periods, reindex=False)
                .variance
                .T
            )

        return conditional_volatility_forecast_df