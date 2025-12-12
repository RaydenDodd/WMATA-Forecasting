#src/model_lightgbm.py
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import shap
import matplotlib.pyplot as plt



from features import add_weather_features, add_weekend_feature, add_hour_features, add_lag_and_rolling_features
from evaluate import evaluate_predictions


def train_lightgbm(train_df, val_df, feature_cols):
    train_x = train_df[feature_cols]
    train_y = train_df["Entries"]
    val_x = val_df[feature_cols]
    val_y = val_df["Entries"]

    cat_cols = [c for c in feature_cols if train_x[c].dtype == "object"]

    for df_ in (train_x, val_x):
        for c in cat_cols:
            df_[c] = df_[c].astype("category")

    lgb_train = lgb.Dataset(train_x, label=train_y, categorical_feature=cat_cols)
    lgb_val   = lgb.Dataset(val_x, label=val_y, reference=lgb_train, categorical_feature=cat_cols)

    params = {
        "objective": "regression",
        "metric": ["mae", "rmse"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    val_pred = np.clip(val_pred, 0, None)
    val_df["Entries_Pred_lgbm"] = val_pred

    rmse = np.sqrt(mean_squared_error(val_y, val_pred))
    mae = mean_absolute_error(val_y, val_pred)
    return model, rmse, mae, val_df


def main():
    print("Loading merged ridership + weather data...")
    df = pd.read_csv("Data/ridership_weather_merged.csv")
    df, weather_columns = add_weather_features(df)
    df, weekend_columns = add_weekend_feature(df)
    df, lag_columns, rolling_mean_columns = add_lag_and_rolling_features(df, lags_hours=(1, 3, 6, 12, 24, 168, 336), windows_hours=(3, 6, 12, 24))
    df, hour_columns = add_hour_features(df)
    num_cols = ["temp", "prcp", "wspd", "rhum", "Hour", "Holiday",] + weather_columns + weekend_columns + lag_columns + hour_columns + rolling_mean_columns

    cat_cols = ["Station", "Year", "DayOfWeek", "ServiceType", "TimePeriod", "DayType", "Month"]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    feature_cols = num_cols + cat_cols
    print(f"Features: {feature_cols}\n")

    # Train on 2012–2018
    train = df[(df["Datetime"] >= "2017-01-01") & (df["Datetime"] <= "2018-12-31")]

    # Test on 2019
    test = df[(df["Datetime"] >= "2019-01-01") & (df["Datetime"] <= "2019-12-31")]

    model, rmse, mae, test_pred = train_lightgbm(train, test, feature_cols)
    print(f"\nLightGBM Test RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    save_path = "Data/lightgbm_model.txt"
    model.save_model(save_path)
    print(f"Saved LightGBM model to {save_path}")
    evaluate_predictions(test_pred, pred="Entries_Pred_lgbm")

    for i in range(10):
        ax = lgb.plot_tree(model, tree_index=i, figsize=(20, 15))
        plt.savefig(f"Plots/tree_{i}.png", dpi=300, bbox_inches="tight")
        plt.close()

    ax = lgb.plot_importance(model, max_num_features=20)
    plt.savefig("Plots/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    X_sample = test[feature_cols].copy()
    explainer = shap.TreeExplainer(model)
    X_sample = X_sample.sample(n=min(1000, len(X_sample)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("Plots/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.show()
if __name__ == "__main__":
    main()