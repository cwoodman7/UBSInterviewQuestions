import numpy as np
import pandas as pd
from adl import ADL
from model import Model
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_data(file):
    """Returns a list containing the complete time-series for each variable."""
    df = pd.read_csv(file, index_col=0)

    #re-write this - make a new df where data are organised in columns. We can then use the row labels to help with
    #identifying missing observations

    indices = np.arange(4)
    series = np.array([df.values[i] for i in indices])
    comp_series = []
    for x, y in enumerate(series):
        indices = [not i for i in np.isnan(y)]
        comp_series.append(y[indices])
    return comp_series


def adf(series, names):
    """Conducts ADF test"""
    for i, s in enumerate(series):
        adf = sm.tsa.adfuller(s, regression="ct", autolag="AIC")
        p_value = list(adf)[1]
        if p_value < 0.1:
            print(f"Reject the null of a unit-root for {names[i]} at the 10% level with a p-value of {p_value}")
        else:
            print(f"Fail to reject the null of a unit-root for {names[i]} at the 10% level with a p-value of {p_value}")


def get_residuals(series, names):
    """Generates the appropriate series with which to identify outliers for each variable. Outliers are identified
    using the log-return series for equity indices and the fitted residuals from the estimation of a pure Autoregressive
    process for the VIX and for LIBOR. The decision to use the fitted residuals from an AR regression for the VIX and
    LIBOR was based on the fact that volatility and interest rates are typically modelled as mean-reverting processes
    with an autoregressive structure e.g., (G)ARCH model for volatility and a Vasicek/OU process for interest rares."""

    returns = []
    for i, s in enumerate(series):
        if i in [0, 1]:
            returns.append(np.log(s[1:]) - np.log(s[:-1]))
        else:
            X_0 = pd.DataFrame(np.zeros(len(s)), columns=['zeros'])
            y = pd.DataFrame(s, columns=[names[i]])
            adl = ADL(X_0, y, p=1, q=0, poly=False)
            beta_hat, fitted_values, IS_statistics = adl.fit_model()
            y = np.array(y).transpose()[0]
            fitted_values = np.array(fitted_values).transpose()
            ys = y[len(y) - len(fitted_values):]
            residuals = ys - fitted_values
            returns.append(residuals)
            Model.clear_fitted_values()

    return returns


def z_score(series):
    """Returns a time-series of z-scores for the series"""
    z = [(series[i] - np.mean(series[i])) / np.std(series[i]) for i in np.arange(len(series))]
    return z


def outliers(returns):
    """This function uses a simple Z-score methodology to identify outliers in each series. The function returns a list
    of lists where each list contains the indices of anomalous observations given the specified value for ALPHA. """
    ALPHA = 0.001
    anomalies = []
    scores = z_score(returns)
    for i, series in enumerate(returns):
        anomaly_list = []
        for index, element in enumerate(series):
            if np.abs(scores[i][index]) > norm.ppf(1 - ALPHA):
                anomaly_list.append(index)
        anomalies.append(anomaly_list)
    return anomalies


def plot_outliers(series, indices, names):
    """Plots outliers"""
    for i, s in enumerate(series):
        t = np.linspace(0, len(s), len(s))
        plt.plot(t, s, '-gD', markevery=list(indices[i]), color=list(mcolors.TABLEAU_COLORS)[0])
        plt.title(f'Time series plot for {names[i]} Returns/Residuals')
        plt.xlabel('Time Period')
        plt.ylabel(f"{names[i]} Residuals")
        plt.legend(['Outliers'])
        plt.show()


def get_quarterly_data(file):
    """This function takes the original input data and returns a quarterly dataset starting from the first date
     at which all series have non-missing values. Missing observations are assumed to be randomly distributed through
     time, such that they are replaced using linear interpolation"""
    data = pd.read_csv(file, index_col=0)
    quarterly_columns = np.array([data.columns[i] if data.columns[i][-1] in ['3', '6', '9']
                                                     or data.columns[i][-2:] in ['12'] else '0'
                                  for i in range(len(data.columns))])
    quarterly_columns = list(quarterly_columns[quarterly_columns != '0'])
    quarterly_data = data[quarterly_columns].iloc[:, 40:]
    quarterly_data = quarterly_data.interpolate(axis=1)
    return quarterly_data


def remove_outliers(series, interpolate=True):
    """Removes outliers from each series that is passed as input in the series list. If interpolate is set to True, then
    the outliers are replaced using linear interpolation, otherwise they are removed from the series. """
    new_series = []
    outlier_indices = outliers(series)
    for i, s in enumerate(series):
        if interpolate:
            c = 1
            new_s = []
            most_recent = None
            for j in range(len(s)):
                if j in outlier_indices[i]:
                    prev_index = None
                    next_index = None
                    if most_recent is not None:
                        prev_index = most_recent
                    k = 1
                    while k < len(s) - 1 - j:
                        if j + k in outlier_indices[i]:
                            k += 1
                        else:
                            next_index = j + k
                            break
                    if prev_index is None:
                        interpolated_value = s[next_index]
                    elif next_index is None:
                        interpolated_value = s[prev_index]
                    else:
                        interpolated_value = s[prev_index] + (c * (s[next_index]-s[prev_index])/(next_index-prev_index))
                        if c == next_index - prev_index - 1:
                            c = 1
                        else:
                            c += 1
                    new_s.append(interpolated_value)
                else:
                    new_s.append(s[j])
                    most_recent = j
            new_series.append(new_s)
        else:
            new_indices = [j for j in np.arange(len(s)) if j not in outlier_indices[i]]
            new_series.append(s[new_indices])

    return new_series


def interpolate_levels(level_series, residual_series):
    """The residual series are those that are used to identify outliers. The function then interpolates the level
    series using the location of the outliers identified."""
    new_levels = []
    outlier_indices = outliers(residual_series)

    for i, s in enumerate(residual_series):
        c = 1
        new_level = [level_series[i][0]]
        most_recent = 0
        for j in range(len(s)):
            if j in outlier_indices[i]:
                next_index = None
                k = 1
                while k < len(s) - 1 - j:
                    if j + k in outlier_indices[i]:
                        k += 1
                    else:
                        next_index = j + k + 1
                        break
                if next_index is None:
                    interpolated_value = level_series[i][most_recent]
                else:
                    interpolated_value = \
                        level_series[i][most_recent] + \
                        (c * (level_series[i][next_index] - level_series[i][most_recent])/(next_index-most_recent))
                    if c == next_index - most_recent - 1:
                        c = 1
                    else:
                        c += 1
                new_level.append(interpolated_value)
            else:
                new_level.append(level_series[i][j+1])
                most_recent = j + 1
        new_levels.append(new_level)

    return new_levels
