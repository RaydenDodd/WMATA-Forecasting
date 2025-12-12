# src/make_weather_csv.py
from datetime import datetime
from meteostat import Point, Hourly
import pandas as pd
from data_prep import print_dataframe_info

start = datetime(2012, 1, 1)
end = datetime(2025, 6, 16)
dc = Point(38.9072, -77.0369)
data = Hourly(dc, start, end)
df = data.fetch()

df = df.reset_index().rename(columns={"time": "Datetime"})
keep_cols = ["Datetime", "temp", "prcp", "wspd", "pres", "rhum"]
df = df[keep_cols]
for col in keep_cols[1:]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

out_path = "Data/weather_hourly_dc.csv"
df.to_csv(out_path, index=False)

print(f"Saved weather to {out_path}")
print_dataframe_info(df)