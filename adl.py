import numpy as np
import pandas as pd
from model import Model
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures


class ADL(Model):
    def __init__(self, X, y, p, q, poly):
        super().__init__(
            X, y, poly
        )
        self.p = p
        self.q = q

        assert type(p).__name__ == 'int', 'Number of endogenous lags must be a non-negative integer'
        assert type(q).__name__ == 'int', 'Number of exogenous lags must be a non-negative integer'

        Model.counter += 1

    def fit_model(self):
        self.X = Model.X_orig.copy()
        if self.poly:
            return self.__fit_non_linear()
        else:
            return self.__fit_linear()

    def predict(self, orig_X, level_X, **kwargs):
        fitted_models = [x[0] for x in Model.fitted_values]
        if self.poly:
            if 'Non Linear ADL' in fitted_models:
                y_hat = self.__predict_non_linear(orig_X, level_X, **kwargs)
            else:
                raise Exception('Must fit model before making predictions')
        else:
            if 'Linear ADL' in fitted_models:
                y_hat = self.__predict_linear(orig_X, level_X, **kwargs)
            else:
                raise Exception('Must fit model before making predictions')
        return y_hat

    def residual_plot(self):
        if self.poly:
            self._Model__resid_plot('Non Linear Static')
        else:
            self._Model__resid_plot('Linear Static')

    def __fit_linear(self):
        """Fits a linear version of the ADL model"""
        self.__add_lags(self.p, self.q)
        ones = np.ones(len(self.X)).reshape(len(self.X), 1)
        self.X.insert(loc=0, column='cons', value=ones)
        beta_hat, fitted_values, IS_statistics = self._Model__fit('Linear ADL')
        return beta_hat, fitted_values, IS_statistics

    def __fit_non_linear(self):
        """Fits a non-linear version of the ADL model"""
        self.__add_lags(self.p, self.q)
        self.polynomial_features()
        beta_hat, fitted_values, IS_statistics = self._Model__fit('Non Linear ADL', reg=False)
        return beta_hat, fitted_values, IS_statistics

    def __predict_linear(self, orig_X, level_X, **kwargs):
        new_X = self.__construct_pred_data(orig_X, level_X, **kwargs)
        ones = np.ones(len(new_X)).reshape(len(new_X), 1)
        new_X.insert(loc=0, column='cons', value=ones)
        regs = np.array(new_X[self.X.columns].iloc[-1])
        fitted_beta = np.array([self.beta[name] for name in self.beta])
        y_hat = np.inner(regs, fitted_beta)
        return y_hat

    def __predict_non_linear(self, orig_X, level_X, **kwargs):
        new_X = self.__construct_pred_data(orig_X, level_X, **kwargs)
        poly = PolynomialFeatures(degree=Model.max_degree, interaction_only=False, include_bias=True)
        poly_features = pd.DataFrame(poly.fit_transform(new_X),
                                     columns=poly.get_feature_names(input_features=new_X.columns))
        poly_features.rename(columns={'1': 'cons'}, inplace=True)
        new_X = poly_features[self.X.columns]
        values = np.array(new_X.iloc[-1, :])
        fitted_beta = np.array([self.beta[name] for name in self.beta])
        y_hat = np.inner(values, fitted_beta)
        return y_hat

    def __construct_pred_data(self, orig_X, level_X, **kwargs):
        new_X = self._Model__update_data(orig_X, level_X, **kwargs)

        Xp = self.X.copy()
        self.X = new_X
        self.__add_lags(0, self.q)
        new_X = self.X.copy()
        self.X = Xp
        y_lags = pd.DataFrame(sm.tsa.add_lag(np.array(Model.y_orig), lags=self.p - 1, drop=False))
        col_names = []
        for i in range(1, self.p + 1):
            col_names.append(f'{self.y.columns[0]} L{i}')
        y_lags.columns = col_names

        if self.p < self.q - 1:
            y_lags.drop(y_lags.head(self.q - 1 - self.p).index, inplace=True)
            y_lags.reset_index(level=0, drop=True, inplace=True)
        elif self.p > self.q - 1:
            new_X.drop(new_X.head(self.p - self.q + 1).index, inplace=True)
            new_X.reset_index(level=0, drop=True, inplace=True)

        new_X = pd.concat([y_lags, new_X], axis=1)

        return new_X
    
    def __add_lags(self, p, q):
        """Modifies the regressors to include p lags of the dependent variable, the contemporaneous (optional) values
          of the independent variables and q-1 lags of each independent variable. Setting q=0 yields a pure AR model"""
        column_names = []
        Xl = pd.DataFrame(sm.tsa.add_lag(np.array(self.y), lags=p, drop=True))
        if p < q - 1:
            Xl.drop(Xl.head(q - 1 - p).index, inplace=True)
        Xl.reset_index(level=0, drop=True, inplace=True)
        if p > 0:
            for i in range(1, p + 1):
                column_names.append(f'{self.y.columns[0]} L{i}')
        if q > 0:
            for i in range(len(self.X.columns)):
                x = np.array(self.X.iloc[:, i])
                new_cols = pd.DataFrame(sm.tsa.add_lag(x, lags=q - 1, drop=False))
                if q - 1 < p:
                    new_cols.drop(new_cols.head(p - q + 1).index, inplace=True)
                    new_cols.reset_index(level=0, drop=True, inplace=True)
                Xl = pd.concat([Xl, new_cols], axis=1)
                for j in range(q):
                    if j == 0:
                        column_names.append(self.X.columns[i])
                    else:
                        column_names.append(f'{self.X.columns[i]} L{j}')

        Xl.columns = column_names
        self.X = Xl
        self.y.drop(self.y.head(np.maximum(p, q - 1)).index, inplace=True)
        self.y.reset_index(level=0, drop=True, inplace=True)
