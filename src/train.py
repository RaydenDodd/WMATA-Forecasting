# src/train_all.py
import os
import pandas as pd
import argparse
import lightgbm as lgb
from prettytable import PrettyTable


from evaluate import mae, rmse, mape, evaluate_predictions
from baseline import baseline_ma_weeks
from model_prophet import  add_prophet_predictions_all_stations
from model_lightgbm import train_lightgbm, predict_lightgbm, save_lgb_plots
from features import add_weather_features, add_weekend_feature, add_lag_and_rolling_features, add_hour_features, train_val_test_split, add_time_features, add_month_features, add_timeperiod_features, print_split_info

from prettytable import PrettyTable

def print_and_save_pretty_results(results_all, results_cut, csv_path_all="Models/model_comparison_all.csv", csv_path_cut="Models/model_comparison_cutoff.csv", cutoff=20):
    results_all.to_csv(csv_path_all, index=False)
    results_cut.to_csv(csv_path_cut, index=False)

    # PrettyTable: ALL
    x_all = PrettyTable()
    x_all.field_names = ["Model", "MAE", "RMSE", "MAPE"]
    for _, r in results_all.iterrows():
        x_all.add_row([
            str(r["model"]),
            f"{float(r['MAE']):,.2f}",
            f"{float(r['RMSE']):,.2f}",
            f"{float(r['MAPE']):,.2f}",
        ])
    print(x_all.get_string(title="Model Comparison (All Test Rows)"))

    x_cut = PrettyTable()
    x_cut.field_names = ["Model", "MAE", "RMSE", "MAPE"]
    for _, r in results_cut.iterrows():
        x_cut.add_row([
            str(r["model"]),
            f"{float(r['MAE']):,.2f}",
            f"{float(r['RMSE']):,.2f}",
            f"{float(r['MAPE']):,.2f}",
        ])
    print(x_cut.get_string(title=f"Model Comparison (Entries ≥ {cutoff})"))

    print(f"\nSaved model comparison tables to {csv_path_all} and {csv_path_cut}")

def main():
    parser = argparse.ArgumentParser(description="Train and compare ridership models")
    parser.add_argument("--use-saved", action="store_true", help="Load saved models/predictions if available instead of retraining")
    args = parser.parse_args()
    use_saved = args.use_saved

    # Load merged dataset
    # ------------------------------------------------------------------
    print("Loading merged ridership + weather data...")
    df = pd.read_csv("Data/ridership_weather_merged.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    train, val, test = train_val_test_split(df, train_end="2018-01-01", val_end="2019-01-01", test_end="2020-01-01")

    print_split_info(train, val, test)

    # Base test frame where we accumulate all model predictions
    test_base = test[["Station", "Datetime", "Entries"]].sort_values("Datetime").copy()

    # BASELINE moving average model
    # ------------------------------------------------------------------
    print("\nRunning Baseline moving-average model...")
    n_weeks = 4
    baseline_file = f"Models/baseline_ma{n_weeks}w.csv"
    if use_saved and os.path.exists(baseline_file):
        print(f"Loading saved baseline predictions from {baseline_file}...")
        df_baseline_all = pd.read_csv(baseline_file)
        df_baseline_all["Datetime"] = pd.to_datetime(df_baseline_all["Datetime"])
        baseline_col = f"Entries_Pred_ma{n_weeks}w"
    else:
        print("Computing baseline moving average predictions from scratch...")
        df_baseline_all, baseline_col = baseline_ma_weeks(df, n_weeks=n_weeks)

    df_baseline_test = df_baseline_all[(df_baseline_all["Datetime"] >= test["Datetime"].min()) & (df_baseline_all["Datetime"] <= test["Datetime"].max())].copy()
    
    evaluate_predictions(df_baseline_test, pred=baseline_col)

    # Align baseline predictions to the test rows using Station + Datetime
    test_base = test_base.merge(df_baseline_test[["Station", "Datetime", baseline_col]], on=["Station", "Datetime"], how="left")
    test_base = test_base.rename(columns={baseline_col: "Entries_Pred_baseline"})


    # PROPHET 
    # ------------------------------------------------------------------
    print("\n=== Running Prophet for all stations ===")
    prophet_file = "Models/prophet.csv"
    if use_saved and os.path.exists(prophet_file):
        print(f"Loading saved Prophet predictions from {prophet_file}...")
        df_prophet_pred = pd.read_csv(prophet_file)
        df_prophet_pred["Datetime"] = pd.to_datetime(df_prophet_pred["Datetime"])
    else:
        print("Training Prophet model from scratch...")
        df_prophet = df.copy()
        df_prophet["Datetime"] = pd.to_datetime(df_prophet["Datetime"])

        # Recreate Prophet feature set
        df_prophet, weather_cols_p = add_weather_features(df_prophet)
        df_prophet, dow_cols       = add_time_features(df_prophet)
        df_prophet, weekend_cols_p = add_weekend_feature(df_prophet)
        df_prophet, month_cols     = add_month_features(df_prophet)
        df_prophet, timeperiod_cols = add_timeperiod_features(df_prophet)

        extra_regressors = [
            "temp", "prcp", "wspd", "rhum", "Hour", "Holiday",
        ] + weather_cols_p + dow_cols + weekend_cols_p + month_cols + timeperiod_cols

        combined_datetimes = pd.concat([train["Datetime"], val["Datetime"]]).unique()
        prophet_train_val = df_prophet[df_prophet["Datetime"].isin(combined_datetimes)]
        prophet_test  = df_prophet[df_prophet["Datetime"].isin(test["Datetime"])]

        df_prophet_pred = add_prophet_predictions_all_stations(
            prophet_train_val,
            prophet_test,
            extra_regressors=extra_regressors,
            use_covid_regressor=False
        )

    print("\nProphet metrics on TEST...")
    evaluate_predictions(df_prophet_pred, pred="Entries_Pred_prophet")

    # Merge Prophet predictions into combined test frame
    test_base = test_base.merge(df_prophet_pred[["Station", "Datetime", "Entries_Pred_prophet"]], on=["Station", "Datetime"], how="left")

    # LIGHTGBM
    # ------------------------------------------------------------------
    print("\nRunning LightGBM...")
    df_lgb = df.copy()
    df_lgb["Datetime"] = pd.to_datetime(df_lgb["Datetime"])

    df_lgb, weather_cols_l = add_weather_features(df_lgb)
    df_lgb, weekend_cols_l = add_weekend_feature(df_lgb)
    df_lgb, lag_cols, rolling_cols = add_lag_and_rolling_features(df_lgb, lags_hours=(1, 3, 6, 12, 24, 168, 336), windows_hours=(3, 6, 12, 24))
    df_lgb, hour_cols = add_hour_features(df_lgb)

    num_cols = [
        "temp", "prcp", "wspd", "rhum", "Hour", "Holiday",
    ] + weather_cols_l + weekend_cols_l + lag_cols + hour_cols + rolling_cols

    cat_cols = ["Station", "Year", "DayOfWeek", "ServiceType", "TimePeriod", "DayType", "Month"]
    for c in cat_cols:
        df_lgb[c] = df_lgb[c].astype("category")

    feature_cols = num_cols + cat_cols
    
    lgb_train = df_lgb[df_lgb["Datetime"].isin(train["Datetime"])].copy()
    lgb_val   = df_lgb[df_lgb["Datetime"].isin(val["Datetime"])].copy()
    lgb_test  = df_lgb[df_lgb["Datetime"].isin(test["Datetime"])].copy()
    lgb_file = "Models/lightgbm_model.txt"
    if use_saved and os.path.exists(lgb_file):
        print(f"Loading saved LightGBM model from {lgb_file}...")
        model_lgb = lgb.Booster(model_file=lgb_file)
        lgb_test_pred = predict_lightgbm(model_lgb, lgb_test, feature_cols, name="test")
        lgb_val_pred = predict_lightgbm(model_lgb, lgb_val, feature_cols, name="val")
    else:
        print("Training LightGBM model from scratch...")

        # Currently train_lightgbm(train_df, val_df, feature_cols) treats val_df as the eval/test set
        model_lgb, lgb_val_pred, lgb_test_pred = train_lightgbm(lgb_train, lgb_val, lgb_test, feature_cols)
    print("\n--- LightGBM metrics on VAL ---")
    evaluate_predictions(lgb_val_pred, pred="Entries_Pred_lgbm")
    print("\n--- LightGBM metrics on TEST ---") 
    evaluate_predictions(lgb_test_pred, pred="Entries_Pred_lgbm")
    save_lgb_plots(model_lgb, lgb_test_pred, feature_cols)
    # Merge LightGBM predictions into combined test frame
    test_base = test_base.merge(lgb_test_pred[["Station", "Datetime", "Entries_Pred_lgbm"]], on=["Station", "Datetime"], how="left",)

    # Final comparison table
    # ------------------------------------------------------------------
    print("\nMODEL COMPARISON ON SHARED TEST ROWS...")

    # Only keep rows where we have all 3 predictions
    comparison_df = test_base.dropna(subset=["Entries_Pred_baseline", "Entries_Pred_prophet", "Entries_Pred_lgbm"]).copy()

    cutoff = 20

    models = [
        ("Baseline", "Entries_Pred_baseline"),
        ("Prophet",  "Entries_Pred_prophet"),
        ("LightGBM", "Entries_Pred_lgbm"),
    ]

    rows_all = []
    for model_name, col in models:
        if col in comparison_df.columns:
            rows_all.append({
                "model": model_name,
                "MAE":  mae(comparison_df["Entries"], comparison_df[col]),
                "RMSE": rmse(comparison_df["Entries"], comparison_df[col]),
                "MAPE": mape(comparison_df["Entries"], comparison_df[col]),
            })
    results_all = pd.DataFrame(rows_all)

    mask = comparison_df["Entries"] >= cutoff
    df_cut = comparison_df.loc[mask]

    rows_cut = []
    for model_name, col in models:
        if col in df_cut.columns and len(df_cut) > 0:
            rows_cut.append({
                "model": model_name,
                "MAE":  mae(df_cut["Entries"], df_cut[col]),
                "RMSE": rmse(df_cut["Entries"], df_cut[col]),
                "MAPE": mape(df_cut["Entries"], df_cut[col]),
            })
    results_cut = pd.DataFrame(rows_cut)
    #PRetty Tables
    print_and_save_pretty_results(results_all, results_cut)

    # Save combined predictions
    comparison_df.to_csv("Data/all_models_test_with_split.csv", index=False)
    print("\nSaved combined predictions to Data/all_models_test_with_split.csv")


if __name__ == "__main__":
    main()
