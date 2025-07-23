import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    # "autogluon/chronos-t5-tiny",
    #  "amazon/chronos-t5-mini"
    #  "amazon/chronos-t5-small"
    "amazon/chronos-t5-base",
    # "amazon/chronos-t5-large"
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# df = pd.read_csv(    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")
df = pd.read_csv("pa_hosps_allData.csv")
# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# print(df)

prediction_length = 12
row_cut = 220

true_vals = df["value"].iloc[row_cut : row_cut + prediction_length]

df = df.iloc[0:row_cut]

# print(df["value"])

context = torch.tensor(df["value"])


forecast = pipeline.predict(
    context, prediction_length
)  # shape [num_series, num_samples, prediction_length]

# visualize the forecast
forecast_index = range(row_cut, row_cut + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)


plt.figure(figsize=(8, 4))
plt.ylim(top=max(true_vals))
plt.plot(df["value"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.plot(
    forecast_index, true_vals, color="green", label="true hosps"
)  # change median, color, label
plt.fill_between(
    forecast_index,
    low,
    high,
    color="tomato",
    alpha=0.3,
    label="80% prediction interval",
)
plt.legend()
plt.grid()
plt.show()
