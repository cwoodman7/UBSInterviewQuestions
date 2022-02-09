import numpy as np
import pandas as pd
from model import Model
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures


class Static(Model):
    def __init__(self, X, y, poly):
        super().__init__(
            X, y, poly
        )
        Model.counter += 1

    def fit_model(self):
        """Fits model"""
        self.X = Model.X_orig.copy()
        if self.poly:
            return self.__fit_non_linear()
        else:
            return self.__fit_linear()

    def predict(self, orig_X, level_X, **kwargs):
        """Returns a 1-step ahead prediction of the target variable"""
        fitted_models = [x[0] for x in Model.fitted_values]
        if self.poly:
            if 'Non Linear Static' in fitted_models:
                y_hat = self.__predict_non_linear(orig_X, level_X, **kwargs)
            else:
                raise Exception('Must fit model before making predictions')
        else:
            if 'Linear Static' in fitted_models:
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
        """Fits a linear version of the static model"""
        ones = np.ones(len(self.X)).reshape(len(self.X), 1)
        self.X.insert(loc=0, column='cons', value=ones)
        beta_hat, fitted_values, IS_statistics = self._Model__fit('Linear Static')
        return beta_hat, fitted_values, IS_statistics

    def __fit_non_linear(self):
        """Fits a non-linear version of the static model"""
        self.polynomial_features()
        beta_hat, fitted_values, IS_statistics = self._Model__fit('Non Linear Static', reg=False)
        return beta_hat, fitted_values, IS_statistics

    def __predict_linear(self, orig_X, level_X, **kwargs):
        new_X = self._Model__update_data(orig_X, level_X, **kwargs)
        ones = np.ones(len(new_X)).reshape(len(new_X), 1)
        new_X.insert(loc=0, column='cons', value=ones)

        values = np.array(new_X[self.X.columns].iloc[-1])
        fitted_beta = np.array([self.beta[name] for name in self.beta])
        y_hat = np.inner(values, fitted_beta)
        return y_hat

    def __predict_non_linear(self, orig_X, level_X, **kwargs):
        new_X = self._Model__update_data(orig_X, level_X, **kwargs)
        poly = PolynomialFeatures(degree=Model.max_degree, interaction_only=False, include_bias=True)
        poly_features = pd.DataFrame(poly.fit_transform(new_X),
                                     columns=poly.get_feature_names(input_features=new_X.columns))
        poly_features.rename(columns={'1': 'cons'}, inplace=True)
        new_X = poly_features[self.X.columns]

        values = np.array(new_X.iloc[-1, :])
        fitted_beta = np.array([self.beta[name] for name in self.beta])
        y_hat = np.inner(values, fitted_beta)
        return y_hat
