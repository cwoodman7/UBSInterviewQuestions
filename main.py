import utils1
import utils2
import numpy as np
import pandas as pd
from adl import ADL
from time import time
from model import Model
from static import Static
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    question1()
    question2()


def question1():
    print("QUESTION 1. \n")

    # a.)
    file = 'data.csv'
    series = utils1.load_data(file)
    names = ['Equity 1', 'Equity 2', 'Vix', 'Libor 1M']

    # b.)
    utils1.adf(series, names)
    returns = utils1.get_residuals(series, names)
    outlier_indices = utils1.outliers(returns)

    print(f'Outlier indices for Equity 1 returns: {outlier_indices[0]}')
    print(f'Outlier indices for Equity 2 returns: {outlier_indices[1]}')
    print(f'Outlier indices for Vix returns: {outlier_indices[2]}')
    print(f'Outlier indices for Libor 1M returns: {outlier_indices[3]}')

    utils1.plot_outliers(returns, outlier_indices, names)

    """Update Series to Remove or Interpolate Outliers"""
    new_returns = utils1.remove_outliers(returns, interpolate=True)

    for i, r in enumerate(new_returns):
        df1 = pd.DataFrame(returns[i], columns=['Original'])
        df2 = pd.DataFrame(r, columns=['Outliers Removed'])
        df = pd.concat([df1, df2])
        plt.plot(df)
        plt.xlabel('Time-Period')
        plt.ylabel(f"{names[i]} Residual")
        plt.legend(['Original', 'Interpolated'])
        plt.title(f'Original Residuals and Residuals with Outliers Removed for {names[i]}')
        plt.show()

    # c.)
    """Compute quarterly series"""
    quarterly_data = utils1.get_quarterly_data(file)
    y = np.array(quarterly_data.loc['VIX Index (index points)'])
    eq1 = np.array(quarterly_data.loc['Equity 1 (index points)'])
    eq2 = np.array(quarterly_data.loc['Equity2 (index points) '])
    libor = np.array(quarterly_data.loc['Libor 1M (in %) '])

    """Remove Outliers"""
    series = [eq1, eq2, y, libor]
    residuals = utils1.get_residuals(series, names)
    outlier_indices = utils1.outliers(residuals)

    print(f'Outlier indices for monthly Equity 1 returns: {outlier_indices[0]}')
    print(f'Outlier indices for monthly Equity 2 returns: {outlier_indices[1]}')
    print(f'Outlier indices for monthly Vix returns: {outlier_indices[2]}')
    print(f'Outlier indices for monthly Libor 1M returns: {outlier_indices[3]}')

    utils1.plot_outliers(residuals, outlier_indices, names)
    new_series = utils1.interpolate_levels(series, residuals)

    for i, s in enumerate(new_series):
        df1 = pd.DataFrame(series[i], columns=['Original'])
        df2 = pd.DataFrame(s, columns=['Outliers Removed'])
        df = pd.concat([df1, df2])
        plt.plot(df)
        plt.xlabel('Time-Period')
        plt.ylabel(f"{names[i]}")
        plt.legend(['Original', 'Interpolated'])
        plt.title(f'Original Series and Series with Outliers Removed for {names[i]}')
        plt.show()

    eq1, eq2, y, libor = new_series

    """Construct dependent and independent variables"""
    y = np.array(y[1:])
    x1 = np.log(eq1[1:]) - np.log(eq1[:-1])
    x2 = np.log(eq2[1:]) - np.log(eq2[:-1])
    x3 = np.array(libor[1:])
    # x3 = np.array(libor_new[1:]) - np.array(libor_new[:-1])

    X = pd.DataFrame(np.stack([x1, x2, x3], axis=1), columns=['eq1', 'eq2', 'libor'])
    y = pd.DataFrame(y, columns=['vix'])

    static = Static(X, y, poly=False)
    poly_static = Static(X, y, poly=True)
    adl = ADL(X, y, 3, 4, poly=False)
    poly_adl = ADL(X, y, 3, 4, poly=True)

    static_beta, static_fitted_values, static_IS_statistics = static.fit_model()
    poly_beta, poly_fitted_values, poly_IS_statistics = poly_static.fit_model()
    ADL_beta, ADL_fitted_values, ADL_IS_statistics = adl.fit_model()
    poly_ADL_beta, poly_ADL_fitted_values, poly_ADL_IS_statistics = poly_adl.fit_model()

    """Plot the distribution of fitted values"""
    Model.dist_plot()

    print("\n")
    print("In-Sample Goodness-Of-Fit Statistics:")
    print(f"IS static model statistics: {static_IS_statistics}")
    print(f"IS poly static model statistics: {poly_IS_statistics}")
    print(f"IS ADL model statistics: {ADL_IS_statistics}")
    print(f"IS poly ADL model statistics: {poly_ADL_IS_statistics}")

    """Check the level of multi-collinearity"""
    # corr = poly_adl.check_collinear()
    # print(corr)

    # d.)
    level_X = pd.DataFrame(np.stack([eq1, eq2, libor], axis=1), columns=['eq1', 'eq2', 'libor'])
    # (i.)
    eq1_val = 1300
    static_y = static.predict(X, level_X, eq1=eq1_val)
    poly_y = poly_static.predict(X, level_X, eq1=eq1_val)
    adl_y = adl.predict(X, level_X, eq1=eq1_val)
    poly_adl_y = poly_adl.predict(X, level_X, eq1=eq1_val)

    print("\n")
    print(f"The estimated value of the VIX conditional on the value of Equity 1 decreasing to {eq1_val} "
          f"is {round(static_y, 2)} according to the static model")
    print(f"The estimated value of the VIX conditional on the value of Equity 1 decreasing to {eq1_val} "
          f"is {round(poly_y, 2)} according to the non-linear static model")
    print(f"The estimated value of the VIX conditional on the value of Equity 1 decreasing to {eq1_val} "
          f"is {round(adl_y, 2)} according to the ADL model")
    print(f"The estimated value of the VIX conditional on the value of Equity 1 decreasing to {eq1_val} "
          f"is {round(poly_adl_y, 2)} according to the non-linear ADL model")

    # (ii.)
    eq2_val = 10000
    static_y = static.predict(X, level_X, eq2=eq2_val)
    poly_y = poly_static.predict(X, level_X, eq2=eq2_val)
    adl_y = adl.predict(X, level_X, eq2=eq2_val)
    poly_adl_y = poly_adl.predict(X, level_X, eq2=eq2_val)

    print("\n")
    print(f"The estimated value of the VIX conditional on the value of Equity 2 increasing to {eq2_val} "
          f"is {round(static_y, 2)} according to the static model")
    print(f"The estimated value of the VIX conditional on the value of Equity 2 increasing to {eq2_val} "
          f"is {round(poly_y, 2)} according to the non-linear static model")
    print(f"The estimated value of the VIX conditional on the value of Equity 2 increasing to {eq2_val} "
          f"is {round(adl_y, 2)} according to the ADL model")
    print(f"The estimated value of the VIX conditional on the value of Equity 2 increasing to {eq2_val} "
          f"is {round(poly_adl_y, 2)} according to the non-linear ADL model")

    print("\n")


def question2():
    print("QUESTION 2. \n")

    # a.)
    N = 10
    run = utils2.runs(N)
    print(f"Random sequence of heads and tails: {utils2.sequence(run)}")

    # b.)
    max_run = utils2.max_runs(run)
    print(f"Longest run of heads: {max_run}")

    # c.)
    N = 1000
    trials = 100000
    dist_vec = np.zeros(N + 1)
    from time import time
    start = time()
    for i in range(trials):
        run = utils2.runs(N)
        max_run = utils2.max_runs(run)
        dist_vec[int(max_run)] += 1
    print(time()-start)

    prob_vec = dist_vec * (1 / np.sum(dist_vec))
    mean = np.inner(prob_vec, np.arange(len(prob_vec)))
    pr = np.sum(prob_vec[7:])

    print(prob_vec[0:40])
    print(dist_vec[0:40])

    # Print results
    print("\n")
    print(f"RESULT FROM SIMULATIONS: \n")
    print(f"Average longest run of heads: {np.round(mean, 3)}")
    print(f"Proportion of longest runs of heads greater than 6: {np.round(pr, 3)}")
    print(f"Most common longest run of heads: {list(dist_vec).index(np.max(dist_vec))}")
    print(f"Shortest run of heads: {np.nonzero(prob_vec)[0][0]}")
    print(f"Longest of run of heads: {np.nonzero(prob_vec)[0][-1]}")

    # Plot frequency distribution
    x_vec = np.arange(N + 1)
    plt.plot(x_vec, dist_vec, '-gD', markevery=[6], color=list(mcolors.TABLEAU_COLORS)[0])
    plt.xlim(0, 50)
    plt.xlabel("Longest Run of Consecutive Heads")
    plt.ylabel("Frequency")
    plt.title(f"Frequency Distribution for the Longest Runs of Heads in Sequences of Length {N}")
    plt.annotate(f'Proportion of Samples where the \n Maximum Run of Heads was 6 is {np.round(prob_vec[6], 3)}',
                 xy=(6, dist_vec[6]), xytext=(30, dist_vec[6] + 100),
                 arrowprops=dict(facecolor=list(mcolors.TABLEAU_COLORS)[0], shrink=0.05, width=2.5))
    plt.show()


if __name__ == '__main__':
    main()
