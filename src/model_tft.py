# src/model_tft.py
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from features import (
    train_val_test_split,
    print_split_info,
    add_weather_features,
    add_weekend_feature,
    add_lag_and_rolling_features,
    add_hour_features,
)
from evaluate import evaluate_predictions


def pick_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    return device


def add_time_idx(df, datetime_col, group_col):
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values([group_col, datetime_col])

    min_dt = df[datetime_col].min()
    df["time_idx"] = ((df[datetime_col] - min_dt).dt.total_seconds() // 3600).astype(int)
    return df, min_dt


def prepare_tft_dataloaders(
    train_df,
    val_df,
    test_df,
    target_col,
    group_col,
    num_cols,
    cat_cols,
    train_end,
    val_end,
    max_encoder_length=168,
    max_prediction_length=24,
    batch_size=64,
):
    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    full_df["Datetime"] = pd.to_datetime(full_df["Datetime"])

    # make sure target is numeric float (fixes "continuous target" warning)
    full_df[target_col] = pd.to_numeric(full_df[target_col], errors="coerce").astype("float32")

    for c in cat_cols:
        full_df[c] = full_df[c].astype("category")

    full_df, global_min_dt = add_time_idx(full_df, datetime_col="Datetime", group_col=group_col)

    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)

    train_mask = full_df["Datetime"] <= train_end
    val_mask = (full_df["Datetime"] > train_end) & (full_df["Datetime"] <= val_end)
    test_mask = full_df["Datetime"] > val_end

    time_varying_known_reals = [c for c in num_cols if c != target_col]
    time_varying_known_categoricals = [c for c in cat_cols if c != group_col]
    static_categoricals = [group_col]

    # allow unseen categories in val/test
    cat_encoders = {c: NaNLabelEncoder(add_nan=True) for c in (static_categoricals + time_varying_known_categoricals)}

    training = TimeSeriesDataSet(
        full_df.loc[train_mask],
        time_idx="time_idx",
        target=target_col,
        group_ids=[group_col],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_encoder_length=1,
        min_prediction_length=1,
        time_varying_unknown_reals=[target_col],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        static_categoricals=static_categoricals,
        categorical_encoders=cat_encoders,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        full_df.loc[val_mask],
        predict=True,
        stop_randomization=True,
    )

    testset = TimeSeriesDataSet.from_dataset(
        training,
        full_df.loc[test_mask],
        predict=True,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=11, persistent_workers=True)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=11, persistent_workers=True)
    test_loader = testset.to_dataloader(train=False, batch_size=batch_size, num_workers=11, persistent_workers=True)

    return training, validation, testset, train_loader, val_loader, test_loader, global_min_dt


def train_tft(training, train_loader, val_loader, device, max_epochs=10):
    accelerator = "gpu" if device == "cuda" else "cpu"

    pl_trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        enable_checkpointing=True,
        log_every_n_steps=50,
        limit_train_batches=0.02,
        limit_val_batches=1.0,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(),
        output_size=7,
        reduce_on_plateau_patience=2,
    )

    pl_trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return tft, pl_trainer


def predict_tft_to_df(
    tft,
    dataset,
    df_out,
    global_min_dt,
    datetime_col="Datetime",
    group_col="Station",
    pred_col="Entries_Pred_tft",
):
    out = tft.predict(dataset, mode="prediction", return_x=True)

    preds = out[0]
    x = out[1]

    g = x["groups"]
    if isinstance(g, (list, tuple)):
        groups = g[0].detach().cpu().numpy().reshape(-1)
    else:
        groups = g.detach().cpu().numpy().reshape(-1)

    pred_df = pd.DataFrame(
        {
            "time_idx": x["decoder_time_idx"].detach().cpu().numpy().reshape(-1),
            group_col: groups,
            pred_col: preds.detach().cpu().numpy().reshape(-1),
        }
    )

    global_min_dt = pd.to_datetime(global_min_dt)
    pred_df[datetime_col] = global_min_dt + pd.to_timedelta(pred_df["time_idx"], unit="h")

    out = df_out.copy()
    out[datetime_col] = pd.to_datetime(out[datetime_col])

    return out.merge(pred_df[[datetime_col, group_col, pred_col]], on=[datetime_col, group_col], how="left")


def main():
    device = pick_device()

    print("Loading merged ridership + weather data...")
    df = pd.read_csv("Data/ridership_weather_merged.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # force target numeric early (also helps feature builders)
    df["Entries"] = pd.to_numeric(df["Entries"], errors="coerce").astype("float32")

    df, weather_columns = add_weather_features(df)
    df, weekend_columns = add_weekend_feature(df)
    df, lag_columns, rolling_mean_columns = add_lag_and_rolling_features(df, lags_hours=(1, 3, 6, 12, 24, 168, 336), windows_hours=(3, 6, 12, 24))  # keep your args
    df, hour_columns = add_hour_features(df)

    num_cols = (
        ["temp", "prcp", "wspd", "rhum", "Hour", "Holiday"]
        + weather_columns
        + weekend_columns
        + lag_columns
        + hour_columns
        + rolling_mean_columns
    )
    cat_cols = ["Station", "DayOfWeek", "ServiceType", "TimePeriod", "DayType", "Month"]

    train_end = "2018-01-01"
    val_end = "2019-01-01"
    test_end = "2020-01-01"

    train_df, val_df, test_df = train_val_test_split(df, train_end=train_end, val_end=val_end, test_end=test_end)
    print_split_info(train_df, val_df, test_df)

    training, validation, testset, train_loader, val_loader, test_loader, global_min_dt = prepare_tft_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col="Entries",
        group_col="Station",
        num_cols=num_cols,
        cat_cols=cat_cols,
        train_end=train_end,
        val_end=val_end,
        max_encoder_length=168,
        max_prediction_length=24,
        batch_size=256,
    )

    tft, _trainer = train_tft(training, train_loader, val_loader, device=device, max_epochs=1)

    df_test_pred = predict_tft_to_df(tft, testset, test_df, global_min_dt, pred_col="Entries_Pred_tft")
    evaluate_predictions(df_test_pred, pred="Entries_Pred_tft")


if __name__ == "__main__":
    main()
