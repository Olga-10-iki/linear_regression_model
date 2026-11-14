# Kenya Market Food Price Predictor

**Mission:** help smallholder farmers get fair prices for their produce.
**Problem:** farmers often sell to middlemen without knowing the going market rate. This project predicts the price per kg
of a food commodity at a Kenyan market for a given month and selling channel (retail/wholesale), so a farmer or cooperative
can decide where and when to sell and recognise an unfair offer.

## Dataset

**WFP Food Prices for Kenya** — World Food Programme market price monitoring, published on the Humanitarian Data Exchange:
<https://data.humdata.org/dataset/wfp-food-prices-for-kenya>. 29,298 rows × 16 columns (2006–2026): date, region, county,
market, coordinates, commodity category, commodity, unit, price type, price flag, currency and price. Prices are converted
to **KES per kg (or litre)**, which is the regression target.

## API (Swagger UI)

Public Swagger UI: **/docs** ← *replace with your Render URL*

- `POST /predict` — 8 inputs (commodity, pricetype, admin1, latitude, longitude, year, month, pack_size) → predicted KES/kg
- `POST /retrain` — upload a CSV in the WFP format (see `summative/API/sample_new_data.csv`); the model is refit on the
  original + new rows and used by `/predict` immediately
- `GET /options` — valid values for the text inputs

*The free Render service sleeps after 15 minutes; the first request can take up to a minute.*

## Video demo

YouTube: **https://youtu.be/REPLACE_ME** ← *replace with your video link*

## How to run the mobile app

1. Install Flutter (3.24 or newer) and set up an Android emulator (Android Studio → Device Manager) or connect a phone with
   USB debugging on. Check with `flutter doctor`.
2. Clone the repository and open the app folder:
   ```bash
   git clone https://github.com/<your-username>/linear_regression_model.git
   cd linear_regression_model/summative/FlutterApp
   ```
3. Generate the platform folders (only needed once if `android/` is missing; existing files are kept):
   ```bash
   flutter create . --project-name food_price_app --platforms android,ios
   ```
4. Install packages: `flutter pub get`
5. Start the emulator (or plug in the phone), then run: `flutter run`
6. Fill in the 8 fields and tap **Predict**. Example: `Beans (dry)`, `Retail`, `North Eastern`, `0.11`, `40.32`, `2025`,
   `6`, `1`.

The app calls the API URL set in `lib/main.dart` (`apiBaseUrl`).

---

### Repository layout

```
linear_regression_model/
├── README.md
└── summative/
    ├── pyproject.toml, uv.lock          # uv project (uv sync)
    ├── linear_regression/multivariate.ipynb
    ├── API/                             # FastAPI service (Render root directory)
    │   ├── prediction.py
    │   ├── requirements.txt             # uv export --format requirements.txt --no-hashes --no-dev
    │   ├── best_model.pkl
    │   ├── wfp_food_prices_ken.csv      # original data, used by /retrain
    │   └── sample_new_data.csv          # example upload for /retrain
    └── FlutterApp/
```

### Run the API locally

```bash
cd summative
uv sync
cd API
uv run uvicorn prediction:app --reload     # http://127.0.0.1:8000/docs
```

### Render settings

| Setting | Value |
|---|---|
| Root directory | `summative/API` |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn prediction:app --host 0.0.0.0 --port $PORT` |
| Environment | `PYTHON_VERSION=3.13.0` (optional `ALLOWED_ORIGINS` for extra CORS origins) |
