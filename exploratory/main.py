# ------------------------------------------------------------
# Setup
# libraries
import os
import sys
import inspect
import logging
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import register_matplotlib_converters
from itertools import product
from multiprocessing import Pool
from functools import partial

# directories
WDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PDIR = os.path.dirname(WDIR)
sys.path.insert(0, PDIR)

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
plt.style.use("seaborn-colorblind")
warnings.filterwarnings("ignore")
register_matplotlib_converters()

print(f"Completed: Environment Setup")


# ------------------------------------------------------------
# EDA
# filter
dt = pd.read_csv(f"{PDIR}/data/crops.csv", sep=",", encoding="cp1252")
df = (
    dt[
        (dt["Area"] == "Australia")
        & (dt["Item"] == "Potatoes")
        & (dt["Unit"] == "hg/ha")
    ]
    .dropna()
    .reset_index()
)
df["Year"] = pd.to_datetime(df["Year"].astype(int).astype(str) + "06", format="%Y%m")

# visualise
fig = plt.figure()
ax = plt.axes()
ax.plot(df["Year"], df["Value"])
plt.show()

# decomposition
ts = df[["Value"]]
dc = sm.tsa.seasonal_decompose(ts, model="additive", freq=12)
dv = dc.plot()

# auto-correlation
fig, ax = plt.subplots(2, 1, figsize=(16, 16))
ac = plot_acf(ts["Value"], ax=ax[0], zero=False)
pc = plot_pacf(ts["Value"], ax=ax[1], zero=False)
plt.show()

print(f"Completed: Time-series Analysis")

# ------------------------------------------------------------
# Model


# grid search
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


# paramaters
fc = pd.DataFrame()
P = D = Q = range(0, 2)
S = 1
pdq = list(product(P, D, Q))
PDQ = [(x[0], x[1], x[2], S) for x in pdq]

# hyperparameter grid search
try:
    with Pool() as pool:
        iterList = product(pdq, PDQ)
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
    predict = best_fit.get_prediction()
    predict_ci = predict.conf_int()

    # set bounds and calculate mean
    predict_ci.columns = ["lb", "ub"]
    predict_ci["mu"] = predict_ci.mean(axis=1)

except Exception as e:
    logger.info(f"{e}: Prediction failed")
    pass

# visualise
cb = df[["Value"]]
cb["Forecast"] = predict_ci["mu"]
cb.plot()

print(f"Completed: Predictions")
