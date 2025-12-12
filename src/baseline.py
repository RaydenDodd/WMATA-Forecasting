#src/baseline.py
import pandas as pd
from evaluate import evaluate_predictions

def baseline_ma_weeks(df, n_weeks=4):
    df = df.sort_values(["Station", "Datetime"]).copy()
    df = df.set_index("Datetime")

    preds = []
    col_name = f"Entries_Pred_ma{n_weeks}w"

    hours_per_week = 7 * 24

    for _station, g in df.groupby("Station"):
        lag_list = []
        for k in range(1, n_weeks + 1):
            lag = g["Entries"].shift(k * hours_per_week)
            lag_list.append(lag)

        stacked = sum(lag_list) / float(n_weeks)

        tmp = g.copy()
        tmp[col_name] = stacked
        preds.append(tmp)

    out = pd.concat(preds).reset_index()
    out = out.dropna(subset=[col_name])
    return out, col_name

def main():
    # Load merged data
    print("Loading merged ridership + weather data...")
    df = pd.read_csv("Data/ridership_weather_merged.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Choose how many weeks to look back
    n_weeks = 124
    print(f"\nRunning {n_weeks}-week moving average baseline...")
    df_pred, pred_col = baseline_ma_weeks(df, n_weeks=n_weeks)

    # Evaluate on 2019 dat
    test_2019 = df_pred[
        (df_pred["Datetime"] >= "2019-01-01") &
        (df_pred["Datetime"] <= "2019-12-31")
    ].copy()

    print(f"\nTest rows for 2019: {len(test_2019)}")
    evaluate_predictions(test_2019, pred=pred_col)

    out_path = f"Data/baseline_ma{n_weeks}w_2019.csv"
    test_2019.to_csv(out_path, index=False)
    print(f"\nSaved 2019 baseline predictions to {out_path}")


if __name__ == "__main__":
    main()