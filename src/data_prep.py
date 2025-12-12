# src/data_prep.py
import pandas as pd
import numpy as np

def load_ridership():
    print("Loading ridership data...")
    df = pd.read_csv("Data/Entries_by_Year_Full_Data_data.csv")

    # Ensure datetime
    df["date_only"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p").dt.date
    df["Datetime"] = pd.to_datetime(df["date_only"].astype(str)) + pd.to_timedelta(df["Hour"].astype(int), unit="h")

    # Keep only relevant columns
    keep_cols = [
        "Datetime",
        "Station Name",
        "Entries",
        "Exits",
        "Day of Week",
        "Holiday",
        "Service Type",
        "Time Period",
        "Hour",
        "Month",
        "DAY_TYPE",
        "Year",
    ]
    df = df[keep_cols].copy()

    # Make some nicer feature names
    df.rename(columns={
        "Station Name": "Station",
        "Day of Week": "DayOfWeek",
        "Service Type": "ServiceType",
        "Time Period": "TimePeriod",
        "DAY_TYPE": "DayType",
    }, inplace=True)
    # binary holiday
    df["Holiday"] = (df["Holiday"].str.lower() == "yes").astype(int)

    # Clean Entries
    df["Entries"] = pd.to_numeric(df["Entries"], errors="coerce")
    df["Entries"] = df["Entries"].replace([np.inf, -np.inf], np.nan)
    df["Entries"] = df["Entries"].clip(lower=0)
    df["Entries"] = df["Entries"].fillna(0)

    return df

def load_weather():
    w = pd.read_csv("Data/weather_hourly_dc.csv")
    w["Datetime"] = pd.to_datetime(w["Datetime"])
    return w

def merge_ridership_weather():
    df = load_ridership()
    w = load_weather()

    # left join on WMATA data
    df_merged = df.merge(w, on="Datetime", how="left")
    return df_merged

def print_dataframe_info(df: pd.DataFrame):
    print("\n" + "-" * 60)
    print("DATAFRAME SHAPE")
    print("-" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n" + "-" * 60)
    print("COLUMNS & DTYPES")
    print("-" * 60)
    print(df.dtypes)

    print("\n" + "-" * 60)
    print("STATS")
    print("-" * 60)
    print(df.describe().T)

    print("\n" + "-" * 60)
    print("SAMPLE ROWS")
    print("-" * 60)
    print(df.head(20))

if __name__ == "__main__":
    # when you run: uv run src/data_prep.py
    merged = merge_ridership_weather()
    out_path = "Data/ridership_weather_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved merged dataset to {out_path}")
    print_dataframe_info(merged)