# src/model_prophet.py

import pandas as pd
from prophet import Prophet
from features import add_weather_features, add_time_features, add_weekend_feature, add_month_features, add_timeperiod_features
from tqdm import tqdm
from joblib import Parallel, delayed
from evaluate import evaluate_predictions
import numpy as np

def add_covid_regressor(df, covid_start="2020-03-15", covid_end="2023-01-01"):
    df = df.copy()
    mask = (df["Datetime"] >= pd.to_datetime(covid_start)) & (df["Datetime"] <= pd.to_datetime(covid_end))
    df["COVID_Lockdown"] = mask.astype(int)
    return df


def train_prophet_for_station(df_station, extra_regressors):
    df_p = df_station.rename(columns={"Datetime": "ds"}).copy()
    df_p["y"] = np.log1p(df_p["Entries"])  # log(1 + Entries)

    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)

    for col in extra_regressors:
        model.add_regressor(col)

    cols_for_fit = ["ds", "y"] + list(extra_regressors)
    model.fit(df_p[cols_for_fit])

    return model


def make_prophet_forecast(model, df_future):
    df_f = df_future.rename(columns={"Datetime": "ds"}).copy()
    forecast = model.predict(df_f)

    df_out = df_future.copy()
    df_out["Entries_Pred_prophet"] = np.expm1(forecast["yhat"].values)
    df_out["Entries_Pred_prophet"] = df_out["Entries_Pred_prophet"].clip(lower=0)
    return df_out

def fit_and_predict_one_station(df_train, df_test, regressors):
    df_train = df_train.sort_values("Datetime").copy()
    df_test = df_test.sort_values("Datetime").copy()
    m = train_prophet_for_station(df_train, extra_regressors=regressors)
    test_with_preds = make_prophet_forecast(m, df_test)
    return test_with_preds

def add_prophet_predictions_all_stations(df_train, df_test, extra_regressors, use_covid_regressor=True, covid_start="2020-03-15", covid_end="2023-01-01"):
    df_train = df_train.copy()

    # Optionally add COVID regressor
    if use_covid_regressor:
        df_train = add_covid_regressor(df_train, covid_start, covid_end)
        df_test = add_covid_regressor(df_test, covid_start, covid_end)
        regressors = list(extra_regressors) + ["COVID_Lockdown"]
    else:
        regressors = list(extra_regressors)

    # Prepare grouped data once
    stations = sorted(set(df_train["Station"]).intersection(df_test["Station"]))
    station_groups = []
    for st in stations:
        g_train = df_train[df_train["Station"] == st]
        g_test = df_test[df_test["Station"] == st]
        if len(g_train) == 0 or len(g_test) == 0:
            continue
        station_groups.append((st, g_train, g_test))
    
    # Parallel over stations
    results = Parallel(n_jobs=-1)(
        delayed(fit_and_predict_one_station)(g_train, g_test, regressors)
        for (_station, g_train, g_test) in tqdm(station_groups, desc="Running Prophet for stations")
    )
    results = [r for r in results if r is not None]

    # Combine all stations back together
    df_test_with_preds = pd.concat(results, axis=0).sort_values(["Station", "Datetime"]).reset_index(drop=True)
    # Save output
    df_test_with_preds.to_csv("Models/prophet.csv", index=False)
    print("\nSaved output to Models/prophet.csv")
    return df_test_with_preds

def main():
    # Setup
    # -----------------------------------------------------------------------------
    print("Loading merged ridership + weather data...")

    df = pd.read_csv("Data/ridership_weather_merged.csv")

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df, weather_columns = add_weather_features(df)
    df, dow_columns = add_time_features(df)
    df, weekend_columns = add_weekend_feature(df)
    df, month_columns = add_month_features(df)
    df, timeperiod_columns = add_timeperiod_features(df)

    extra_regressors = ["temp", "prcp", "wspd", "rhum", "Hour", "Holiday",] + weather_columns + dow_columns+ weekend_columns+ month_columns + timeperiod_columns
    print(f"Using extra regressors: {extra_regressors}\n")

    # Train on 2012–2018
    train = df[(df["Datetime"] >= "2012-01-01") & (df["Datetime"] <= "2018-12-31")]

    # Test on 2019
    test = df[(df["Datetime"] >= "2019-01-01") & (df["Datetime"] <= "2019-12-31")]
    # One Station Metrics
    # -----------------------------------------------------------------------------
    print("Running Prediction for 1 station...")
    station_name="Metro Center"
    train_one = train[train["Station"] == station_name].sort_values("Datetime")
    test_one = test[test["Station"] == station_name].sort_values("Datetime")

    print(f"\nTraining rows: {len(train)}, Testing rows: {len(test)}")
    print(f"Station tested: {station_name}")

    test_with_preds = fit_and_predict_one_station(train_one, test_one, extra_regressors)

    evaluate_predictions(test_with_preds, "Entries_Pred_prophet")

    # All Stations Predictions
    # -----------------------------------------------------------------------------
    print("Running Prophet for all stations...")

    df_pred = add_prophet_predictions_all_stations(train, test, extra_regressors=extra_regressors, use_covid_regressor=True,)

    evaluate_predictions(df_pred, "Entries_Pred_prophet")

if __name__ == "__main__":
    main()