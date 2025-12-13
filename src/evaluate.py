#src/evaluate.py
import numpy as np

# Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Root Mean Squared Error
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Mean Absolute Percentage Error
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def evaluate_predictions(df_test, pred="Entries_Pred"):
    station_mae = mae(df_test["Entries"].values, df_test[pred].values)
    station_rmse = rmse(df_test["Entries"].values, df_test[pred].values)
    station_mape = mape(df_test["Entries"].values, df_test[pred].values)

    print("\nError Metrics")
    print(f"MAE:   {station_mae:.2f}")
    print(f"RMSE:  {station_rmse:.2f}")
    print(f"MAPE:  {station_mape:.2f}%")

    mask = df_test["Entries"] >= 20
    filtered = df_test[mask]

    f_mae = mae(filtered["Entries"].values, filtered[pred].values)
    f_rmse = rmse(filtered["Entries"].values, filtered[pred].values)
    f_mape = mape(filtered["Entries"].values, filtered[pred].values)

    print("\nError Metrics Entries Larger than 20")
    print(f"MAE:   {f_mae:.2f}")
    print(f"RMSE:  {f_rmse:.2f}")
    print(f"MAPE:  {f_mape:.2f}%")
    
    print("\nSample Predictions (True vs Predicted)")
    print(df_test[["Datetime", "Entries", pred]].head(20))
