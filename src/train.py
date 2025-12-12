#src/train.py
import pandas as pd
from evaluate import mae, rmse, mape

rows = []
for model_name, col in [
    ("Baseline", "Entries_Pred_baseline"),
    ("Prophet", "Entries_Pred_prophet"),
    ("LightGBM", "Entries_Pred_lgbm"),
    ("TFT", "Entries_Pred_tft"),
]:
    if col in df.columns:
        rows.append({
            "model": model_name,
            "MAE": mae(df["y"], df[col]),
            "RMSE": rmse(df["y"], df[col]),
            "MAPE": mape(df["y"], df[col]),
        })
results = pd.DataFrame(rows)
print(results)