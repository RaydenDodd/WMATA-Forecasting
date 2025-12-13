# src/data_prep.py
import pandas as pd
import numpy as np

def load_ridership():
    print("Loading ridership data...")
    df = pd.read_csv("Data/Entries_by_Year_Full_Data_data.csv")

    # Parse date and build hourly Datetime
    df["date_only"] = pd.to_datetime(
        df["Date"],
        format="%m/%d/%Y %I:%M:%S %p"
    ).dt.date
    df["Datetime"] = pd.to_datetime(df["date_only"].astype(str)) + pd.to_timedelta(df["Hour"].astype(int), unit="h")

    keep_cols = [
        "Datetime",
        "date_only",
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

    # Nicer names
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

    # 9 AM is in both AM Peak and Midday periods; collapse these into just AM Peak
    # ------------------------------------------------------------------
    mask_9 = df["Hour"] == 9

    if mask_9.any():
        df_9 = df[mask_9].copy()
        df_not_9 = df[~mask_9].copy()

        # Group by Station + date_only + Hour and sum Entries/Exits
        grouped_9 = (
            df_9
            .groupby(["Station", "date_only", "Hour"], as_index=False)
            .agg({
                "Datetime": "first",
                "Entries": "sum",
                "Exits": "sum",
                "DayOfWeek": "first",
                "Holiday": "max",
                "ServiceType": "first",
                "Month": "first",
                "DayType": "first",
                "Year": "first",
            })
        )

        # Force TimePeriod to AM Peak for these combined rows
        grouped_9["TimePeriod"] = "AM Peak (Open-9:30am)"

        cols_order = df.columns
        df = pd.concat([df_not_9, grouped_9[cols_order]], ignore_index=True)

    df = df.drop(columns=["date_only"])

    print("Duplicates after collapse (Station + Datetime):", df.duplicated(subset=["Station", "Datetime"]).sum())

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
    merged = merge_ridership_weather()
    out_path = "Data/ridership_weather_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved merged dataset to {out_path}")
    print_dataframe_info(merged)