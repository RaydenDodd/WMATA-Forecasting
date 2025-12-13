# src/features.py
import pandas as pd
import numpy as np

def train_val_test_split(df, train_end, val_end, test_end=None):
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    if test_end is None:
        test_end = df["Datetime"].max()

    train = df[df["Datetime"] <= train_end].copy()
    val   = df[(df["Datetime"] > train_end) & (df["Datetime"] <= val_end)].copy()
    test  = df[(df["Datetime"] > val_end) & (df["Datetime"] <= test_end)].copy()

    return train, val, test

def print_split_info(train, val, test):
    train_start = train["Datetime"].min()
    train_end   = train["Datetime"].max()

    val_start = val["Datetime"].min()
    val_end   = val["Datetime"].max()

    test_start = test["Datetime"].min()
    test_end   = test["Datetime"].max()

    print(f"Train: {len(train):,} rows: {train_start} - {train_end}")
    print(f"Val:   {len(val):,} rows: {val_start} - {val_end}")
    print(f"Test:  {len(test):,} rows: {test_start} - {test_end}")


def add_weather_features(df):
    df = df.copy()
    new = pd.DataFrame(index=df.index)

    new["Is_Rain"]     = (df["prcp"] > 0).astype(int)
    new["Is_Freezing"] = (df["temp"] <= 0).astype(int)
    new["Is_Snow"]     = ((df["prcp"] > 0) & (df["temp"] <= 1)).astype(int)
    new["FeelsLike"]   = df["temp"] - 0.7 * df["wspd"]
    new["Is_Hot"]      = (df["temp"] >= 30).astype(int)
    result = pd.concat([df, new], axis=1)

    return result, list(new.columns) 

def add_time_features(df):
    df = df.copy()
    dow = pd.get_dummies(df["DayOfWeek"], prefix="DOW")
    df = pd.concat([df, dow], axis=1)
    return df, list(dow.columns)

def add_weekend_feature(df):
    df = df.copy()
    new = pd.DataFrame(index=df.index)
    new["Is_Weekend"] = df["DayType"].isin(["Saturday", "Sunday"]).astype(int)
    result = pd.concat([df, new], axis=1)
    return result, list(new.columns)

def add_month_features(df):
    df = df.copy()
    month = pd.get_dummies(df["Month"], prefix="Month")
    result = pd.concat([df, month], axis=1)
    return result, list(month.columns)

def add_timeperiod_features(df):
    df = df.copy()
    tp = pd.get_dummies(df["TimePeriod"], prefix="TP")
    result = pd.concat([df, tp], axis=1)
    return result, list(tp.columns)

def add_lag_and_rolling_features(df, lags_hours=(1,24,168,336), windows_hours=(3,6,12,24,168,336)):
    df = df.sort_values(["Station","Datetime"]).copy()
    lag_cols = []
    for lag in lags_hours:
        col = f"lag_{lag}h"; df[col] = df.groupby("Station")["Entries"].shift(lag); lag_cols.append(col)
    roll_cols = []
    for w in windows_hours:
        col = f"roll_mean_{w}h"; df[col] = df.groupby("Station")["Entries"].shift(1).rolling(w).mean(); roll_cols.append(col)
    all_cols = lag_cols + roll_cols
    df = df.dropna(subset=all_cols).reset_index(drop=True)
    return df, lag_cols, roll_cols

def add_hour_features(df):
    df = df.copy()
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    return df, ["Hour_sin", "Hour_cos"]