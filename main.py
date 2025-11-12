# main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Security
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
from datetime import date

# Import all your modules
from app import models, schemas, database, locations, weather_fetcher
from app.database import engine, get_db
from app.ml_predictor import ml_predictor
from app.notifications import HIGH_RISK_WARNING_BODY, dispatch_alerts

# --- Day 5: Import Security Dependency ---
from app.dependencies import PROTECTED

# Create all database tables (runs on startup)
try:
    models.Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
except Exception as e:
    print(f"Error creating database tables: {e}")

app = FastAPI(
    title="Elephant Conflict Early Warning System API",
    description="API for predicting human-elephant conflict and managing alerts.",
    version="1.0.0",
    docs_url=None, # Disable the /docs endpoint
    redoc_url=None # Disable the /redoc endpoint
)

# --- Root Endpoint ---
@app.get("/", dependencies=[PROTECTED])
def read_root():
    return {
        "message": "Elephant Conflict Early Warning System API",
        "status": "active",
        "version": "1.0.0"
    }

# --- Day 1: Conflict Incident Endpoints ---

@app.post("/incidents/", response_model=schemas.ConflictIncident, tags=["Incidents"], dependencies=[PROTECTED])
def create_incident(incident: schemas.ConflictIncidentCreate, db: Session = Depends(get_db)):
    db_incident = models.ConflictIncident(**incident.dict())
    db.add(db_incident)
    db.commit()
    db.refresh(db_incident)
    return db_incident

@app.get("/incidents/", response_model=List[schemas.ConflictIncident], tags=["Incidents"], dependencies=[PROTECTED])
def read_incidents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    incidents = db.query(models.ConflictIncident).offset(skip).limit(limit).all()
    return incidents

# --- Day 1: Environmental Data Endpoints ---

@app.post("/environmental-data/", response_model=schemas.EnvironmentalData, tags=["Environment"], dependencies=[PROTECTED])
def create_environmental_data(data: schemas.EnvironmentalDataCreate, db: Session = Depends(get_db)):
    db_data = models.EnvironmentalData(**data.dict())
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data

@app.get("/environmental-data/", response_model=List[schemas.EnvironmentalData], tags=["Environment"], dependencies=[PROTECTED])
def read_environmental_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    data = db.query(models.EnvironmentalData).offset(skip).limit(limit).all()
    return data

# --- Day 2: Machine Learning & Analytics Endpoints ---

@app.post("/train-model/", tags=["Machine Learning"], dependencies=[PROTECTED])
def train_model_endpoint(db: Session = Depends(get_db)):
    print("Received request to train model...")
    result = ml_predictor.train(db)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/predict-risk/", tags=["Machine Learning"], dependencies=[PROTECTED])
def predict_risk_endpoint(
    risk_data: schemas.EnvironmentalDataCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # Convert Pydantic schema to a SQLAlchemy model instance for the predictor
    env_data_model = models.EnvironmentalData(**risk_data.dict())
    prediction = ml_predictor.predict_single_location(db, env_data_model)
    
    if prediction["risk_level"] == "High":
        db_alert = models.Alert(
            location=prediction["location"],
            risk_level=prediction["risk_level"],
            message=HIGH_RISK_WARNING_BODY
        )
        db.add(db_alert)
        db.commit()
        
        background_tasks.add_task(dispatch_alerts, location=prediction["location"])
        prediction["alert_status"] = "Alerts triggered"
    else:
        prediction["alert_status"] = "No alert"
    
    return prediction

@app.get("/risk-heatmap/", tags=["Machine Learning"], dependencies=[PROTECTED])
def get_risk_heatmap(db: Session = Depends(get_db)):
    predictions = ml_predictor.predict_heatmap(db)
    return predictions

@app.get("/analytics/", tags=["Analytics"], dependencies=[PROTECTED])
def get_analytics(db: Session = Depends(get_db)):
    incidents_df = pd.read_sql(db.query(models.ConflictIncident).statement, db.bind)
    if incidents_df.empty:
        return {"error": "No incident data available for analytics."}
        
    incidents_df['timestamp'] = pd.to_datetime(incidents_df['timestamp'])
    
    by_type = incidents_df['incident_type'].value_counts().reset_index()
    by_type.columns = ['type', 'count']
    
    by_location = incidents_df['location'].value_counts().reset_index()
    by_location.columns = ['location', 'count']
    
    incidents_df['month_year'] = incidents_df['timestamp'].dt.to_period('M').astype(str)
    over_time = incidents_df.groupby('month_year').size().reset_index(name='count')
    
    return {
        "by_type": by_type.to_dict('records'),
        "by_location": by_location.to_dict('records'),
        "over_time": over_time.to_dict('records')
    }

# --- Day 3: Farmer Reporting Endpoints ---

@app.post("/report-sighting/", response_model=schemas.FarmerReport, tags=["Farmer Reports"], dependencies=[PROTECTED])
def create_farmer_report(report: schemas.FarmerReportCreate, db: Session = Depends(get_db)):
    db_report = models.FarmerReport(**report.dict())
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

@app.get("/report-sighting/", response_model=List[schemas.FarmerReport], tags=["Farmer Reports"], dependencies=[PROTECTED])
def get_farmer_reports(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    reports = db.query(models.FarmerReport).offset(skip).limit(limit).all()
    return reports

# --- Day 4: 5-Day Forecast Endpoint ---

@app.get("/predict-forecast/{location}", tags=["Forecast"], dependencies=[PROTECTED])
async def get_risk_forecast(location: str, db: Session = Depends(get_db)):
    coords = locations.get_coords(location)
    if not coords:
        raise HTTPException(status_code=404, detail="Location not found")
    
    lat, lon = coords
    
    print(f"Fetching 5-day forecast for {location}...")
    forecast_days = await weather_fetcher.get_weather_forecast(lat, lon)
    
    if not forecast_days:
        raise HTTPException(status_code=500, detail="Could not fetch weather data")
    
    print(f"Got {len(forecast_days)} forecast days.")
    
    latest_env_data = db.query(models.EnvironmentalData).filter(
        models.EnvironmentalData.location == location
    ).order_by(models.EnvironmentalData.date.desc()).first()
    
    current_veg_index = 0.5 
    if latest_env_data:
        current_veg_index = latest_env_data.vegetation_index
    
    risk_forecast_list = []
    for day in forecast_days:
        mock_env_data = models.EnvironmentalData(
            date=day.date,
            location=location,
            rainfall_mm=day.rainfall_mm,
            vegetation_index=current_veg_index
        )
        
        prediction = ml_predictor.predict_single_location(db, mock_env_data)
        prediction['date'] = day.date.isoformat()
        risk_forecast_list.append(prediction)
        
    return risk_forecast_list