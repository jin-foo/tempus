import logging
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import product
from multiprocessing import Pool
from functools import partial
from pandas.plotting import register_matplotlib_converters

# logging
logging.basicConfig(
    filename="monitor.log",
    filemode="w",
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)
logger = logging.getLogger()

# quality-of-life
pd.set_option("display.expand_frame_repr", False)
plt.rcParams["figure.figsize"] = (16, 16)
plt.style.use("seaborn")
warnings.filterwarnings("ignore")
register_matplotlib_converters()


def sorter(df_, country, crop):
    """
    :dataframe df:  complete dataset of crop yields over time
    :return:        filtered dataset by country and crop
    """
    _ = (
        df_[(df_["Area"] == country) & (df_["Item"] == crop) & (df_["Unit"] == "hg/ha")]
        .dropna()
        .reset_index()
    )
    _["Year"] = pd.to_datetime(_["Year"].astype(int).astype(str) + "06", format="%Y%m")
    return _


def grid_builder(ts, coeff):
    """
    Find pairs of pdq & PDQ which minimise AIC for best results
    :param ts:      time-series
    :param coeff:   pairs of (p, d, q) and (P, D, Q) coefficients
    :return:        coefficients which generate the lowest AIC
    """
    try:
        model = sm.tsa.statespace.SARIMAX(
            ts,
            order=coeff[0],
            seasonal_order=coeff[1],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=0)
        return fitted.aic, coeff[0], coeff[1]
    except Exception as e:
        logger.info(f"{e}: Model failed to converge")
        pass


def time_machine(ts, start=0, s=1, dynamic=True):
    """
    Generate predicted values using the best model identified
    :param ts:      time-series values
    :param start:   Zero-indexed observation from which to begin predictions
    :param s:       Seasonal periodicity
    :return:        Dataframe of mean predicted values with confidence intervals
    """
    # paramaters
    P = D = Q = range(0, 2)
    pdq = list(product(P, D, Q))
    pdq_ = [(x[0], x[1], x[2], s) for x in pdq]

    # hyperparameter grid search
    # TODO add get_context(Spawn)
    try:
        with Pool() as pool:
            iterList = product(pdq, pdq_)
            build = partial(grid_builder, ts)
            result = pool.map(build, iterList)
            min_key = min(list(result))

        # set best model
        best_model = sm.tsa.statespace.SARIMAX(
            ts, order=min_key[1], seasonal_order=min_key[2]
        )
        best_fit = best_model.fit()
        print(best_fit.summary())
        logger.info(best_fit.summary())

        try:
            best_fit.plot_diagnostics()
        except Exception as e:
            logger.info(f"{e}: Diagnostics failed")
            pass

        # generate predictions
        predict = best_fit.get_prediction(start=start, dynamic=dynamic)
        predict_ci = predict.conf_int()

        # set bounds and calculate mean
        predict_ci.columns = ["lb", "ub"]
        predict_ci["mu"] = predict.predicted_mean
        return predict_ci

    except Exception as e:
        logger.info(f"{e}: Prediction failed")
        pass


def viewer(df, predict_ci):
    """
    Visualise predictions with confidence interval and error summary
    :df:        Dataframe with observations, predictions and confidence intervals
    :return:    Visual of predictions vs observations with error summary
    """
    # extract
    er = df.dropna()

    # errors
    er["MAE"] = er["mu"] - er["Value"]
    er["MAE"] = er["MAE"].abs()
    er["MSE"] = er["mu"] ** 2 - er["Value"] ** 2
    er["RMSE"] = er["MSE"] ** 0.5
    er["MAPE"] = er["MAE"] / er["Value"]
    MAE = er["MAE"].mean()
    RMSE = er["RMSE"].mean()
    MAPE = er["MAPE"].mean()

    # visualise
    fig, ax = plt.subplots()
    df.plot(
        x="Year Code",
        y="Value",
        kind="scatter",
        ax=ax,
        color="blue",
        label="Observation",
        ylim=0,
    )
    df.plot(
        x="Year Code",
        y="mu",
        kind="line",
        ax=ax,
        color="red",
        label="Prediction",
        ylim=0,
    )
    plt.fill_between(
        df["Year Code"][predict_ci.index],
        df["lb"][predict_ci.index],
        df["ub"][predict_ci.index],
        color="grey",
        alpha=0.25,
    )
    plt.fill_betweenx(
        ax.get_ylim(),
        df["Year Code"][predict_ci.index].min(),
        df["Year Code"][predict_ci.index].max(),
        alpha=0.1,
        zorder=-1,
    )
    plt.text(
        x=2003,
        y=50000,
        s=f"MAE: {{{MAE:.3f}}}\n" f"RMSE: {{{RMSE:.3f}}}\n" f"MAPE: {{{MAPE:.3%}}}\n",
    )
    plt.show()
