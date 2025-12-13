# Forecasting WMATA Metro Ridership  
**Author:** Rayden Dodd  
**Course:** ECE 5424: Statistical Learning (Fall 2024)

This project forecasts hourly station level ridership for the WMATA Metrorail system using historical entries data merged with hourly weather features.  
Models included in this repository:

- **Baseline moving average**
- **Prophet**
- **LightGBM**
- *(Optional)* Temporal Fusion Transformer (not used in the final paper)

All models use a pre-COVID time-based split (train: 2012–2018, validation: 2018, test: 2019).

---

## How to Use

### 1. Initial Setup

1. Install `uv`  
   Verify installation:
   ```bash
   uv --version
   ```

2. Install all dependencies:
   ```bash
   uv sync
   ```

3. Add required data to the `Data/` folder:
   - `Entries_by_Year_Full_Data_data.csv`
   - `weather_hourly_dc.csv`

> **NOTE:** Always run Python scripts via `uv`, for example:  
> `uv run python file.py`

---

## Running the Pipeline

### Step 1 — Generate Weather CSV  
Creates or updates `weather_hourly_dc.csv`:
```bash
uv run python ./src/make_weather_csv.py
```

### Step 2 — Prepare & Merge Data  
Cleans ridership CSV, merges weather, and writes processed output:
```bash
uv run python ./src/data_prep.py
```

---

## Running Individual Models

### Baseline Model
```bash
uv run python ./src/baseline.py
```

### Prophet
```bash
uv run python ./src/model_prophet.py
```

### LightGBM
```bash
uv run python ./src/model_lightgbm.py
```

---

## Training All Models Together

### Train all models
```bash
uv run python ./src/train.py
```

### Use previously saved model outputs
```bash
uv run python ./src/train.py --use-saved
```


---

## Acknowledgments

Ridership data provided by the **WMATA Metrorail Ridership Data Portal**:  
🔗 https://www.wmata.com/initiatives/ridership-portal/
