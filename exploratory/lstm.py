# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
# libraries
import os
import sys
import inspect
import logging
import warnings

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters

# directories
WDIR = f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}"
PDIR = os.path.dirname(WDIR)
sys.path.insert(0, f"{WDIR}/tempus/exploratory")

from tardis import sorter

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
# Prepare
# ------------------------------------------------------------
# ETL
dt = pd.read_csv(f"{WDIR}/tempus/data/crops.csv", sep=",", encoding="cp1252")
df = sorter(dt, "Australia", "Cauliflowers and broccoli")

# scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[["Value"]])
df_ = scaler.transform(df[["Value"]])

# split
size = round(len(df_) * 0.75)
train = df_[:size]
test = df_[size:]

print(f"Completed: Data Preparation")

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------


def sequencer(dataset, rewind):
    t0, t1 = [], []
    for i in range(len(dataset) - rewind):
        t0.append(dataset[i : (i + rewind), 0])
        t1.append(dataset[i + rewind, 0])
    return np.array(t0), np.array(t1)


# create (samples, features)
rewind = 2
train0, train1 = sequencer(train, rewind)
test0, test1 = sequencer(test, rewind)

# reshape (samples, time-step, features)
train0 = np.reshape(train0, (train0.shape[0], 1, train0.shape[1]))
test0 = np.reshape(test0, (test0.shape[0], 1, test0.shape[1]))

# model
model = Sequential()
model.add(LSTM(4, input_shape=(1, rewind)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(train0, train1, epochs=100, batch_size=1, verbose=2)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
# predict
train_predict = model.predict(train0)
test_predict = model.predict(test0)

# invert
train_predict = scaler.inverse_transform(train_predict)
train1 = scaler.inverse_transform([train1])
test_predict = scaler.inverse_transform(test_predict)
test1 = scaler.inverse_transform([test1])

# error
trainScore = math.sqrt(mean_squared_error(train1[0], train_predict[:, 0]))
print(f"Train Score: RMSE {trainScore:.2f}")
testScore = math.sqrt(mean_squared_error(test1[0], test_predict[:, 0]))
print(f"Test Score: %.2f RMSE {testScore:.2f}")

# time-shift
trainPredictPlot = np.empty_like(df_)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[rewind : len(train_predict) + rewind, :] = train_predict
testPredictPlot = np.empty_like(df_)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (rewind * 2) : len(df_), :] = test_predict

# visualise
plt.scatter(x=df["Year Code"], y=scaler.inverse_transform(df_), color="blue", label="Observation")
plt.plot(df["Year Code"], trainPredictPlot, color="black", label="training")
plt.plot(df["Year Code"], testPredictPlot, color="red", label="testing")
plt.legend()
plt.show()
