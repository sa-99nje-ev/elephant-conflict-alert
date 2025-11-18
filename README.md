🐘 Elephant Conflict Early Warning System (EWS)

A full-stack geospatial, analytics-driven system designed to predict, analyze, and visualize human–elephant conflict risk in Sri Lanka.
Built with FastAPI + Streamlit + SQLAlchemy + GIS tools, this project provides:

✔ Conflict risk forecasting
✔ Territory & movement analysis
✔ Terrain corridor modeling
✔ Farmer sighting data ingestion
✔ Fully interactive dashboards
✔ Multi-language alerts (Sinhala, Tamil, English)
✔ Explainability (model-free analytics)
✔ NO third-party SMS dependencies

✨ Key Features
📍 1. Live Conflict Risk Map

Displays real-time model predictions as a geospatial heatmap using Folium.

📊 2. Advanced Analytics Dashboard

Includes:

Incident type distribution

Monthly trends

District heatmaps

Peak hours

Elephant group size analysis

Incident vs district correlation

Mutual information–based conflict driver analysis

🐘 3. Territory Analysis

Extracts hidden patterns from conflict data:

DBSCAN clustering

Seasonal activity cycles

Movement tracking

Territory expansion detection

🗻 4. Terrain Modeling & Elephant Corridors

Using elevation raster (GeoTIFF):

Elevation model

Slope maps

Smoothed terrain

Corridor (valley) extraction using slope thresholds

Lightweight 3D-ready visualizations

🔍 5. Explainability Without a Model

A model-free SHAP-inspired module:

Correlation heatmaps

Sensitivity-based feature importance

Partial dependency style plots

Mutual information driver ranking

No ML model? No problem — this works purely on real conflict data.

🔮 6. 5-Day Forecast & Multi-language Alerts

A demo forecast simulator generating alerts in:

English

Sinhalese

Tamil

Alert types:

Low risk

Medium risk

High risk

With language-switching and message export (JSON).
⚠ No SMS sending backend (Twilio removed entirely).

👨‍🌾 7. Farmer Sighting Reports

Front-end UI + backend endpoint to log real-time elephant sightings:

Location

Coordinates

Elephant count

Behavior

Notes

All saved in the SQLite database.

⚙️ 8. Manual Predict UI

Allows users to manually submit:

Location

Coordinates

Elephant count

Rainfall

Crop type

And get a simulated risk prediction.

💰 9. Economic Impact Dashboard + Causal Inference

Includes:

Expected loss estimation

Resource allocation map

Severity matrix

Causal effect estimation (ATE) to analyze impact of elephant activity

💻 Tech Stack
Backend

FastAPI
SQLAlchemy
SQLite
Pydantic
AioHTTP (for future async APIs)

Frontend

Streamlit
Plotly
Folium
streamlit-folium

Data Science

Pandas
Numpy
Scikit-learn
Joblib
Rasterio
Gaussian filters


📁 Project Structure

elephant-conflict-alert/
│
├── app/
│   ├── __init__.py
│   ├── database.py              # Database engine + session
│   ├── models.py                # SQLAlchemy ORM models
│   ├── schemas.py               # Pydantic schemas
│   ├── locations.py             # Predefined SL conflict hotspot coordinates
│   ├── data_loader.py           # Loads/cleans CSV dataset
│   ├── notifications.py         # Multi-language message builders (no SMS backend)
│   ├── weather_fetcher.py       # (Optional) Weather API async fetcher
│   ├── ml_predictor.py          # Risk prediction logic (rule-based / ML-ready)
│   │
│   ├── herd_analyzer.py         # DBSCAN territory clustering + movement analysis
│   ├── terrain_analyzer.py      # Slope, elevation, corridor extraction
│   ├── explainability.py        # Model-free explainability utilities
│   ├── severity_predictor.py    # Economic loss + severity scoring
│   │
│   ├── test_weather.py          # Utility tester for weather fetching
│   └── main.py                  # FastAPI backend application (core API)
│
├── app/data/
│   ├── sri_lanka_elephant_conflict.csv
│   ├── sri_lanka_elevation.tif
│   ├── elevation_tiles/
│   │   ├── N06E080.hgt
│   │   ├── N06E081.hgt
│   │   ├── N07E080.hgt
│   │   ├── N07E081.hgt
│   │   ├── N08E080.hgt
│   │   ├── N09E080.hgt
│   │   └── ... (raw SRTM tiles)
│
├── generate_elevation.py         # Merges SRTM tiles → final GeoTIFF
├── merge_srtm_tiles.py           # Raw tile stitching helper
├── import_conflicts.py           # Imports CSV into DB cleanly
│
├── dashboard.py                  # Full Streamlit UI (9 modules)
├── elephant_conflict.db          # SQLite database
│
├── requirements.txt
├── .env
├── .gitignore
├── README.md
├── run_day1.py                   # Initialize DB, load data
└── run_day2.py                   # (Optional) Train ML (if enabled)


🚀 How to Run

1️⃣ Install dependencies

pip install -r requirements.txt


2️⃣ Initialize the Database

python run_day1.py


Start FastAPI Backend

uvicorn main:app --reload


Backend docs available at:
👉 http://localhost:8000/docs


Start Streamlit Dashboard

streamlit run dashboard.py


Dashboard opens at:
👉 http://localhost:8501



🔑 Environment Variables

.env file required only for app security:

APP_API_KEY="my-secret-key"


✔ Status

This system is fully functional, error-free, and dashboard-ready.