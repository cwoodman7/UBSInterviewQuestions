import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Model:
    SL = 0.1
    counter = 0
    X_orig = []
    y_orig = []
    max_degree = 3
    fitted_values = []

    @classmethod
    def clear_fitted_values(cls):
        Model.fitted_values = []

    @classmethod
    def dist_plot(cls):
        """Plots the in-sample density of actual and fitted values"""
        assert len(Model.fitted_values) < 10, 'Maximum number of series that can be plotted is 10'
        colours = list(mcolors.TABLEAU_COLORS)
        ax1 = sns.distplot(Model.y_orig, color=colours[0], hist=False, label="Actual values")
        count = 1
        for series in Model.fitted_values:
            colour = colours[count]
            sns.distplot(series[1], color=colour, hist=False, label=f"{series[0]} Fitted values", ax=ax1)
            count += 1
        plt.title("Density Plot of the Fitted and Actual Values of the VIX")
        plt.xlabel("VIX value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def __init__(self, X, y, poly):
        self.X = X.copy()
        self.y = y.copy()
        self.poly = poly
        self.beta = None

        assert type(X).__name__ == 'DataFrame', 'Independent variables must be stored in a Pandas DataFrame'
        assert type(y).__name__ == 'DataFrame', 'Dependent variables must be stored in a Pandas DataFrame'
        assert type(poly).__name__ == 'bool', 'Poly attribute must be a Boolean value'

        Model.X_orig = X.copy()
        Model.y_orig = y.copy()

    def check_collinear(self):
        """Plots the correlation matrix between regressors and returns the numerical matrix in a Pandas DataFrame"""
        corr_matrix = self.X.corr()
        corr_matrix.drop(corr_matrix.head(1).index, inplace=True)
        corr_matrix = corr_matrix.iloc[:, 1:]
        sns.heatmap(corr_matrix, cmap="RdBu")
        plt.show()
        return corr_matrix

    def remove_insignificant_regressors(self):
        """Iteratively removes the regressor with the least significant estimated regression coefficient until all
        coefficients are significant at the SL % significance level. Returns the reduced set of regressors. """
        max_p = 1
        while max_p > Model.SL:
            results = sm.OLS(self.y, self.X).fit(cov_type='HAC',
                                                 cov_kwds={'maxlags': int(np.floor(len(self.y) ** (1 / 4)))})
            p_values = results.pvalues
            max_p = np.max(p_values[1:])
            selection_vector = deque()
            selection_vector.append(1)
            selection_vector.extend([int(p_values[i] < np.maximum(max_p, Model.SL)) for i in range(1, len(p_values))])
            selection_vector = list(selection_vector)
            selected_indices = np.argwhere(np.array(selection_vector) == 1)
            indices = list(selected_indices.reshape(1, -1)[0])
            self.X = self.X.iloc[:, indices]

    def polynomial_features(self):
        """Performs a polynomial expansion to the input feature set. The degree of the expansion is a hyper-parameter
        that is tuned using a 1-dimensional grid search. Returns the expanded feature set."""
        folds = 5
        hyper_params = [{'poly__degree': list(np.arange(1, Model.max_degree + 1))}]
        steps = [('poly', PolynomialFeatures(interaction_only=False, include_bias=True)),
                 ('model', LinearRegression(fit_intercept=False))]
        pipe = Pipeline(steps, verbose=False)
        grid = GridSearchCV(pipe, hyper_params, cv=folds, scoring='r2')  # neg_mean_squared_error
        grid.fit(self.X, self.y)

        name, best_fitted_poly = grid.best_estimator_.__dict__['steps'][0]
        poly_names = best_fitted_poly.get_feature_names(input_features=self.X.columns)
        poly_names[0] = 'cons'
        self.X = best_fitted_poly.transform(self.X)
        self.X = pd.DataFrame(self.X, columns=poly_names)

    def __fit(self, name, reg=False):
        """Fits the model using OLS with HAC standard errors. Returns estimated parameter vector, fitted values and
        in-sample goodness-of-fit statistics."""
        self.remove_insignificant_regressors()
        results = sm.OLS(self.y, self.X).fit(cov_type='HAC',
                                             cov_kwds={'maxlags': int(np.floor(len(self.y) ** (1 / 4)))})
        if reg:
            results = sm.OLS(self.y, self.X).fit_regularized(alpha=0.001, L1_wt=0)
        fitted_values = results.predict(self.X)
        fitted_models = [x[0] for x in Model.fitted_values]
        if name not in fitted_models:
            Model.fitted_values.append((name, fitted_values))
        else:
            index = np.argwhere(np.array([int(i == name) for i in fitted_models]) == 1)[0][0]
            Model.fitted_values[index] = (name, fitted_values)
        beta_hat = dict()
        for index, name in enumerate(self.X.columns):
            beta_hat[f'beta_{name}'] = results.params[index]
        self.beta = beta_hat
        if reg:
            RMSE, R_Squared, Adj_R_Squared = self.__compute_gof()
            IS_statistics = {'RMSE': np.round(RMSE, 3), 'R-Squared': np.round(R_Squared, 3),
                             'Adj R-Squared': np.round(Adj_R_Squared, 3)}
        else:
            print(results.summary())
            IS_statistics = {'RMSE': np.round(results.mse_resid, 3), 'R-Squared': np.round(results.rsquared, 3),
                             'Adj R-Squared': np.round(results.rsquared_adj, 3)}
        return beta_hat, fitted_values, IS_statistics

    def __update_data(self, orig_X, level_X, **kwargs):
        new_row = pd.DataFrame(np.zeros(len(orig_X.columns)).reshape(1, len(orig_X.columns)), columns=orig_X.columns)
        new_X = orig_X.append(new_row, ignore_index=True)
        for col in new_X:
            if col in kwargs:
                if col in ['eq1', 'eq2']:
                    value = np.log(kwargs[col]) - np.log(level_X[col].iloc[-1])
                else:
                    value = kwargs[col]  # - level_X[col].iloc[-1] using levels for vix
                new_X[col].iloc[-1] = value
            else:
                new_X[col].iloc[-1] = np.mean(orig_X[col])
        return new_X

    def __compute_gof(self):
        beta_hat = np.array([self.beta[name] for name in self.beta])
        N = len(np.array(self.X))
        K = len(beta_hat) - 1
        y_hat = np.dot(np.array(self.X), beta_hat)
        y = np.array(Model.y_orig).transpose()[0]
        y = y[len(Model.y_orig) - len(y_hat):]
        y_bar = np.mean(y)
        RSS = np.sum((y - y_hat) ** 2)
        ESS = np.sum((y_hat - y_bar) ** 2)
        TSS = RSS + ESS
        MSE = np.average(np.sum((y - y_hat) ** 2))
        RMSE = np.sqrt(MSE)
        R_Squared = 1 - (RSS / TSS)
        Adj_R_Squared = 1 - (RSS / (N - K - 1) / (TSS / N - 1))
        return RMSE, R_Squared, Adj_R_Squared

    def __resid_plot(self, name):
        y_hat = Model.fitted_values[name]
        y = np.array(Model.y_orig).transpose()[0]
        plt.scatter(y_hat, (y - y_hat))
        plt.show()
