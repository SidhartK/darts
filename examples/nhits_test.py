import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score
from darts.datasets import EnergyDataset
from darts.models.forecasting.nhits import _GType

def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None, filename="nhits_test_image.png"):
    plt.figure(figsize=(8, 5))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(
        "R2: {}".format(r2_score(ts_transformed.univariate_component(0), pred_series))
    )
    plt.legend()
    plt.savefig(filename)

if __name__ == '__main__':
    df = EnergyDataset().load().pd_dataframe()
    df_day_avg = df.groupby(df.index.astype(str).str.split(" ").str[0]).mean().reset_index()
    filler = MissingValuesFiller()
    scaler = Scaler()
    series = scaler.fit_transform(
        filler.transform(
            TimeSeries.from_dataframe(
                df_day_avg, "time", ["generation hydro run-of-river and poundage"]
            )
        )
    ).astype(np.float32)

    train, val = series.split_after(pd.Timestamp("20170901"))

    model_nhits = NHiTSModel(
        input_chunk_length = 30,
        output_chunk_length = 14,
        num_stacks=3,
        num_blocks=1,
        num_layers=4,
        layer_widths=64,
        g_types=[_GType.TREND, _GType.SEASONALITY, _GType.GENERIC],
        pooling_kernel_sizes=[[6], [3], [1]],
        backcast_downsample_freqs=(6, 3, 1),
        forecast_downsample_freqs=(2, 2, 1),
        pooling_layer_name = "Conv1d",
        n_epochs=100,
        nr_epochs_val_period=1,
        batch_size=800,
        model_name="nhits_run",
    )
    model_nhits.fit(train, val_series=val, verbose=True)

    pred_series = model_nhits.predict(n=14, series=series[:-14])
    # pred_series = model_nbeats.historical_forecasts(
    #     series,
    #     start=pd.Timestamp("20170901"),
    #     forecast_horizon=7,
    #     stride=5,
    #     retrain=False,
    #     verbose=True,
    # )
    display_forecast(pred_series, series, "7 day", start_date=pd.Timestamp("20170901"))
    # df["generation hydro run-of-river and poundage"].plot()
    # plt.title("Hourly generation hydro run-of-river and poundage")
