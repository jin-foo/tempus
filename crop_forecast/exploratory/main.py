# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
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

# directories
WDIR = f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}"
PDIR = os.path.dirname(WDIR)
sys.path.insert(0, f"{WDIR}/tempus/exploratory")

from tardis import sorter
from tardis import time_machine
from tardis import viewer

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
plt.rcParams["figure.figsize"] = (16, 8)
plt.style.use("seaborn")
warnings.filterwarnings("ignore")
register_matplotlib_converters()

print(f"Completed: Environment Setup")

# ------------------------------------------------------------
# EDA
# ------------------------------------------------------------
# ETL
dt = pd.read_csv(f"{WDIR}/tempus/data/crops.csv", sep=",", encoding="cp1252")
df = sorter(dt, "Australia", "Cauliflowers and broccoli")

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
fig, ax = plt.subplots(2, 1)
ac = plot_acf(ts["Value"], ax=ax[0])
pc = plot_pacf(ts["Value"], ax=ax[1])
plt.show()

print(f"Completed: Time-series Analysis")

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
# Step-by-Step
# predictions
predict_ci = time_machine(ts, start=0, s=1, dynamic=False)

# visualise
cb = df[["Year Code", "Value"]].merge(
    predict_ci, how="left", left_on=df.index, right_on=predict_ci.index
)
cb = cb.iloc[1:]
viewer(cb, predict_ci)

print(f"Completed: Predictions")