"""
Kenya market food price API.

Predicts the price (KES per kg, or per litre for liquids) of a food commodity at a Kenyan market, so smallholder
farmers can judge where and when to sell. Model: random forest pipeline saved from linear_regression/multivariate.ipynb.

Run locally:   uvicorn prediction:app --reload
Swagger UI:    http://127.0.0.1:8000/docs
"""

import io
import os
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
TRAINING_DATA_PATH = BASE_DIR / "wfp_food_prices_ken.csv"

# ---------------------------------------------------------------------------
# Feature engineering (identical to the notebook)
# ---------------------------------------------------------------------------
UNIT_TO_KG = {"KG": 1, "90 KG": 90, "50 KG": 50, "64 KG": 64, "13 KG": 13, "126 KG": 126,
              "200 G": 0.2, "400 G": 0.4, "L": 1, "500 ML": 0.5}
CATEGORICAL = ["commodity", "pricetype", "admin1"]
NUMERIC = ["latitude", "longitude", "year", "month", "pack_size"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price_per_kg"
RAW_COLUMNS = ["date", "admin1", "latitude", "longitude", "category", "commodity", "unit", "pricetype", "price"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw WFP rows into model-ready rows."""
    df = df[df["unit"].isin(UNIT_TO_KG.keys()) & (df["category"] != "non-food")].copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = dates.dt.year
    df["month"] = dates.dt.month
    df["pack_size"] = df["unit"].map(UNIT_TO_KG)
    df[TARGET] = pd.to_numeric(df["price"], errors="coerce") / df["pack_size"]
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df[TARGET] > 0]
    df[["year", "month"]] = df[["year", "month"]].astype(int)
    return df[FEATURES + [TARGET]].reset_index(drop=True)


def make_model() -> TransformedTargetRegressor:
    """Same pipeline as the best model in the notebook: one-hot -> scale -> random forest, on log(price)."""
    preprocess = Pipeline([
        ("encode", ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
            remainder="passthrough")),
        ("scale", StandardScaler()),
    ])
    forest = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, n_jobs=-1, random_state=42)
    return TransformedTargetRegressor(regressor=Pipeline([("preprocess", preprocess), ("model", forest)]),
                                      func=np.log, inverse_func=np.exp, check_inverse=False)


# ---------------------------------------------------------------------------
# State: the model is loaded from the repository at startup and swapped in memory after /retrain
# ---------------------------------------------------------------------------
model_lock = threading.Lock()
model = joblib.load(MODEL_PATH)
training_data = clean(pd.read_csv(TRAINING_DATA_PATH))

VALID = {col: sorted(training_data[col].unique().tolist()) for col in CATEGORICAL}


def predict_price(row: dict) -> float:
    """Predict KES per kg for one input row (same function as in the notebook, using the in-memory model)."""
    with model_lock:
        current = model
    return float(current.predict(pd.DataFrame([row], columns=FEATURES))[0])


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kenya Market Food Price Predictor",
    description="Predicts the market price of a food commodity in Kenya (KES per kg / litre) to help smallholder "
                "farmers decide where and when to sell. Data: WFP Kenya food prices 2006-2026.",
    version="1.0.0",
)

# CORS: which *websites* (browser origins) may call this API.
# - ALLOWED: only the origins listed below — this service's own Render URL (Swagger UI page) and the local Flutter-web /
#   dev-server addresses I use while developing. Extra origins can be added through the ALLOWED_ORIGINS env var on
#   Render (comma-separated) without changing the code.
# - RESTRICTED: no wildcard "*", so a random website cannot make visitors' browsers call the API (and use its compute)
#   on its behalf. Only GET and POST are allowed because those are the only methods the API exposes, and only the
#   Content-Type / Accept headers the clients actually send.
# - allow_credentials=False: the API uses no cookies or auth headers, so browsers never need to send credentials.
# The Flutter mobile app is not a browser, so CORS does not apply to it — it can call the API from any device.
ALLOWED_ORIGINS = [
    "",   # <- replace with your Render service URL
    "http://localhost:5000",                         # flutter run -d chrome --web-port 5000
    "http://127.0.0.1:5000",
    "http://localhost:8000",                         # local uvicorn (Swagger UI)
    "http://127.0.0.1:8000",
] + [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PriceInput(BaseModel):
    # strict=True: "2024" (a string) is rejected for an int field, 5 (a number) is rejected for a text field, etc.
    model_config = ConfigDict(
        strict=True,
        json_schema_extra={"example": {
            "commodity": "Beans (dry)", "pricetype": "Retail", "admin1": "North Eastern",
            "latitude": 0.11, "longitude": 40.32, "year": 2025, "month": 6, "pack_size": 1.0}},
    )

    commodity: str = Field(..., min_length=2, max_length=60, description="Commodity name exactly as in the WFP data, e.g. 'Maize (white)', 'Beans (dry)'")
    pricetype: str = Field(..., description="'Retail' or 'Wholesale'")
    admin1: str = Field(..., description="Former province / region, e.g. 'Rift Valley', 'Nairobi', 'Coast'")
    latitude: float = Field(..., ge=-4.8, le=5.0, description="Market latitude (Kenya spans about -4.7 to 5.0)")
    longitude: float = Field(..., ge=33.9, le=42.0, description="Market longitude (Kenya spans about 33.9 to 41.9)")
    year: int = Field(..., ge=2006, le=2030, description="Year of sale (data covers 2006-2026)")
    month: int = Field(..., ge=1, le=12, description="Month of sale, 1-12")
    pack_size: float = Field(..., gt=0, le=130, description="Quantity sold in kg (or litres), e.g. 1, 50, 90")

        @field_validator("commodity", "pricetype", "admin1")
    @classmethod
    def must_be_known_category(cls, value: str, info):
        allowed = VALID[info.field_name]
        # ignore capital letters and extra spaces: "beans (dry)" -> "Beans (dry)"
        lookup = {v.lower(): v for v in allowed}
        match = lookup.get(value.strip().lower())
        if match is None:
            raise ValueError(f"'{value}' is not a known {info.field_name}. Allowed values: {', '.join(allowed)}")
        return match


class PriceOutput(BaseModel):
    predicted_price_per_kg: float
    currency: str = "KES"
    unit: str = "per kg (per litre for liquids)"
    predicted_total_price: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["info"])
def root():
    return {"message": "Kenya food price API. See /docs for the Swagger UI.", "training_rows": len(training_data)}


@app.get("/options", tags=["info"])
def options():
    """Valid values for the text inputs."""
    return VALID


@app.post("/predict", response_model=PriceOutput, tags=["model"])
def predict(payload: PriceInput):
    price = predict_price(payload.model_dump())
    return PriceOutput(predicted_price_per_kg=round(price, 2),
                       predicted_total_price=round(price * payload.pack_size, 2))


@app.post("/retrain", tags=["model"])
async def retrain(file: UploadFile = File(..., description="CSV with the same columns as the WFP training data")):
    """Upload new WFP-format rows; the model is refit on original + new data and used immediately by /predict."""
    global model, training_data, VALID

    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        new_raw = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    missing = [c for c in RAW_COLUMNS if c not in new_raw.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV is missing required columns: {missing}")

    new_rows = clean(new_raw)
    if new_rows.empty:
        raise HTTPException(status_code=422, detail="No usable rows after cleaning (check units, dates and prices).")

    # error of the current model on the new rows, before retraining
    with model_lock:
        old_model = model
    mse_before = mean_squared_error(np.log(new_rows[TARGET]), np.log(old_model.predict(new_rows[FEATURES])))

    # best model is a random forest (not SGD), so refit on original + new rows
    combined = pd.concat([training_data, new_rows], ignore_index=True)
    new_model = make_model().fit(combined[FEATURES], combined[TARGET])
    mse_after = mean_squared_error(np.log(new_rows[TARGET]), np.log(new_model.predict(new_rows[FEATURES])))

    with model_lock:
        model = new_model
        training_data = combined
        VALID = {col: sorted(combined[col].unique().tolist()) for col in CATEGORICAL}
    # Kept in memory only: best_model.pkl in the repo stays the notebook's model, and Render's free plan would lose a
    # rewritten file on restart anyway.

    return {
        "message": "Model retrained and now in use by /predict.",
        "rows_uploaded": int(len(new_raw)),
        "rows_used_after_cleaning": int(len(new_rows)),
        "total_training_rows": int(len(combined)),
        "mse_on_new_rows_before (log scale)": round(float(mse_before), 4),
        "mse_on_new_rows_after (log scale)": round(float(mse_after), 4),
    }
