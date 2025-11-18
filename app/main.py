from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
from datetime import date
from app.weather_fetcher import get_weather_forecast

# Import all your modules
from app import models, schemas, database, locations, weather_fetcher
from app.database import engine, get_db
from app.ml_predictor import ml_predictor
from app.notifications import HIGH_RISK_WARNING_BODY, dispatch_alerts

# --- Day 5: Import Security Dependency (COMMENTED OUT FOR DEMO) ---
# from app.dependencies import PROTECTED

# Create all database tables (runs on startup)
try:
    models.Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully.")
except Exception as e:
    print(f"❌ Error creating database tables: {e}")

app = FastAPI(
    title="🐘 Elephant Conflict Early Warning System API",
    description="API for predicting human-elephant conflict and managing alerts. Security temporarily disabled for demo.",
    version="2.0.0"
)

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {
        "message": "🐘 Elephant Conflict Early Warning System API",
        "status": "active", 
        "version": "2.0.0",
        "security": "temporarily_disabled_for_demo",
        "endpoints_available": [
            "/incidents/", "/environmental-data/", "/train-model/", 
            "/predict-risk/", "/risk-heatmap/", "/analytics/",
            "/report-sighting/", "/predict-forecast/{location}"
        ]
    }

# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()}

# --- Day 1: Conflict Incident Endpoints ---

@app.post("/incidents/", response_model=schemas.ConflictIncident, tags=["Incidents"])
def create_incident(incident: schemas.ConflictIncidentCreate, db: Session = Depends(get_db)):
    """Create a new conflict incident report"""
    db_incident = models.ConflictIncident(**incident.dict())
    db.add(db_incident)
    db.commit()
    db.refresh(db_incident)
    return db_incident

@app.get("/incidents/", response_model=List[schemas.ConflictIncident], tags=["Incidents"])
def read_incidents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all conflict incidents with pagination"""
    incidents = db.query(models.ConflictIncident).offset(skip).limit(limit).all()
    return incidents

@app.get("/incidents/stats", tags=["Incidents"])
def get_incident_stats(db: Session = Depends(get_db)):
    """Get incident statistics"""
    total_incidents = db.query(models.ConflictIncident).count()
    incidents_by_type = db.query(
        models.ConflictIncident.incident_type,
        func.count(models.ConflictIncident.id)
    ).group_by(models.ConflictIncident.incident_type).all()
    
    return {
        "total_incidents": total_incidents,
        "by_type": {inc_type: count for inc_type, count in incidents_by_type},
        "locations_covered": db.query(models.ConflictIncident.location).distinct().count()
    }

# --- Day 1: Environmental Data Endpoints ---

@app.post("/environmental-data/", response_model=schemas.EnvironmentalData, tags=["Environment"])
def create_environmental_data(data: schemas.EnvironmentalDataCreate, db: Session = Depends(get_db)):
    """Add new environmental data"""
    db_data = models.EnvironmentalData(**data.dict())
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data

@app.get("/environmental-data/", response_model=List[schemas.EnvironmentalData], tags=["Environment"])
def read_environmental_data(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all environmental data with pagination"""
    data = db.query(models.EnvironmentalData).offset(skip).limit(limit).all()
    return data

@app.get("/environmental-data/latest", tags=["Environment"])
def get_latest_environmental_data(db: Session = Depends(get_db)):
    """Get latest environmental data for each location"""
    from sqlalchemy import desc
    
    locations_data = {}
    for location in locations.get_location_names():
        latest = db.query(models.EnvironmentalData).filter(
            models.EnvironmentalData.location == location
        ).order_by(desc(models.EnvironmentalData.date)).first()
        
        if latest:
            locations_data[location] = {
                "date": latest.date.isoformat(),
                "rainfall_mm": latest.rainfall_mm,
                "vegetation_index": latest.vegetation_index
            }
    
    return locations_data

# --- Day 2: Machine Learning & Analytics Endpoints ---

@app.post("/train-model/", tags=["Machine Learning"])
def train_model_endpoint(db: Session = Depends(get_db)):
    """Train the ML model with current data"""
    print("🎯 Received request to train model...")
    result = ml_predictor.train(db)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    
    return {
        **result,
        "training_time": pd.Timestamp.now().isoformat(),
        "data_points_used": db.query(models.ConflictIncident).count()
    }

@app.post("/predict-risk/", tags=["Machine Learning"])
def predict_risk_endpoint(
    risk_data: schemas.EnvironmentalDataCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Predict conflict risk for given environmental conditions"""
    # Convert Pydantic schema to SQLAlchemy model instance
    env_data_model = models.EnvironmentalData(**risk_data.dict())
    
    # Get prediction
    prediction = ml_predictor.predict_single_location(db, env_data_model)
    
    # Trigger alerts for high risk
    if prediction["risk_level"] == "High":
        db_alert = models.Alert(
            location=prediction["location"],
            risk_level=prediction["risk_level"],
            message=HIGH_RISK_WARNING_BODY
        )
        db.add(db_alert)
        db.commit()
        
        # Dispatch real alerts in background
        background_tasks.add_task(dispatch_alerts, location=prediction["location"])
        prediction["alert_status"] = "🚨 Alerts triggered and queued for delivery"
    else:
        prediction["alert_status"] = "✅ No alert needed"
    
    # Add additional context
    prediction.update({
        "prediction_time": pd.Timestamp.now().isoformat(),
        "model_used": prediction.get("prediction_model", "Unknown"),
        "input_data": {
            "location": risk_data.location,
            "date": risk_data.date.isoformat(),
            "rainfall_mm": risk_data.rainfall_mm,
            "vegetation_index": risk_data.vegetation_index
        }
    })
    
    return prediction

@app.get("/predict-risk/batch", tags=["Machine Learning"])
def predict_risk_batch(db: Session = Depends(get_db)):
    """Predict risk for all locations with latest environmental data"""
    predictions = ml_predictor.predict_heatmap(db)
    
    return {
        "predictions": predictions,
        "total_locations": len(predictions),
        "high_risk_count": len([p for p in predictions if p["risk_level"] == "High"]),
        "generated_at": pd.Timestamp.now().isoformat()
    }

@app.get("/risk-heatmap/", tags=["Machine Learning"])
def get_risk_heatmap(db: Session = Depends(get_db)):
    """Get risk predictions for dashboard heatmap"""
    predictions = ml_predictor.predict_heatmap(db)
    return predictions

@app.get("/analytics/", tags=["Analytics"])
def get_analytics(db: Session = Depends(get_db)):
    """Get comprehensive analytics data"""
    incidents_df = pd.read_sql(db.query(models.ConflictIncident).statement, db.bind)
    
    if incidents_df.empty:
        return {"error": "No incident data available for analytics."}
        
    incidents_df['timestamp'] = pd.to_datetime(incidents_df['timestamp'])
    
    # Incident type analysis
    by_type = incidents_df['incident_type'].value_counts().reset_index()
    by_type.columns = ['type', 'count']
    
    # Location analysis
    by_location = incidents_df['location'].value_counts().reset_index()
    by_location.columns = ['location', 'count']
    
    # Time series analysis
    incidents_df['month_year'] = incidents_df['timestamp'].dt.to_period('M').astype(str)
    over_time = incidents_df.groupby('month_year').size().reset_index(name='count')
    
    # Seasonal patterns
    incidents_df['month'] = incidents_df['timestamp'].dt.month
    seasonal = incidents_df.groupby('month').size().reset_index(name='count')
    
    return {
        "summary": {
            "total_incidents": len(incidents_df),
            "time_period": {
                "first_incident": incidents_df['timestamp'].min().isoformat(),
                "last_incident": incidents_df['timestamp'].max().isoformat()
            },
            "locations_with_incidents": incidents_df['location'].nunique()
        },
        "by_type": by_type.to_dict('records'),
        "by_location": by_location.to_dict('records'),
        "over_time": over_time.to_dict('records'),
        "seasonal_patterns": seasonal.to_dict('records')
    }

@app.get("/system-status", tags=["Analytics"])
def get_system_status(db: Session = Depends(get_db)):
    """Get overall system status and statistics"""
    return {
        "database": {
            "incidents_count": db.query(models.ConflictIncident).count(),
            "environmental_data_count": db.query(models.EnvironmentalData).count(),
            "alerts_count": db.query(models.Alert).count(),
            "farmer_reports_count": db.query(models.FarmerReport).count()
        },
        "ml_model": {
            "is_trained": ml_predictor.is_trained,
            "model_type": "RandomForest" if ml_predictor.is_trained else "RuleBased",
            "features_used": ml_predictor.feature_columns if hasattr(ml_predictor, 'feature_columns') else []
        },
        "locations": {
            "monitored_locations": locations.get_location_names(),
            "total_locations": len(locations.get_location_names())
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }

# --- Day 3: Farmer Reporting Endpoints ---

@app.post("/report-sighting/", response_model=schemas.FarmerReport, tags=["Farmer Reports"])
def create_farmer_report(report: schemas.FarmerReportCreate, db: Session = Depends(get_db)):
    """Submit a new elephant sighting report from farmers"""
    db_report = models.FarmerReport(**report.dict())
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    
    # Log the report
    print(f"📝 New farmer report: {report.elephant_count} elephants in {report.location}")
    
    return db_report

@app.get("/report-sighting/", response_model=List[schemas.FarmerReport], tags=["Farmer Reports"])
def get_farmer_reports(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all farmer sighting reports"""
    reports = db.query(models.FarmerReport).offset(skip).limit(limit).all()
    return reports

@app.get("/report-sighting/stats", tags=["Farmer Reports"])
def get_farmer_report_stats(db: Session = Depends(get_db)):
    """Get statistics about farmer reports"""
    total_reports = db.query(models.FarmerReport).count()
    reports_by_location = db.query(
        models.FarmerReport.location,
        func.count(models.FarmerReport.id)
    ).group_by(models.FarmerReport.location).all()
    
    avg_elephants = db.query(func.avg(models.FarmerReport.elephant_count)).scalar() or 0
    
    return {
        "total_reports": total_reports,
        "average_elephants_per_report": round(avg_elephants, 1),
        "reports_by_location": {loc: count for loc, count in reports_by_location},
        "community_engagement_score": min(total_reports * 10, 100)  # Simple metric
    }

# --- Day 4: 5-Day Forecast Endpoint ---


@app.get("/predict-forecast/{location}", tags=["Forecast"])
async def get_risk_forecast(location: str, db: Session = Depends(get_db)):
    """Get 5-day risk forecast for a specific location using WeatherAPI.com"""
    
    print(f"🎯 Generating 5-day risk forecast for: {location}")
    
    # Get weather forecast using location name (WeatherAPI accepts city names)
    forecast_days = await weather_fetcher.get_weather_forecast(location)
    
    if not forecast_days:
        raise HTTPException(status_code=500, detail="Could not fetch weather forecast data")
    
    print(f"✅ Retrieved {len(forecast_days)} weather forecast days")
    
    # Get current vegetation index for the location
    latest_env_data = db.query(models.EnvironmentalData).filter(
        models.EnvironmentalData.location == location
    ).order_by(models.EnvironmentalData.date.desc()).first()
    
    current_veg_index = latest_env_data.vegetation_index if latest_env_data else 0.5
    
    # Get coordinates for the location
    coords = locations.get_coords(location)
    
    # Generate risk predictions for each forecast day
    risk_forecast_list = []
    high_risk_days = 0
    
    for day in forecast_days:
        # Create environmental data for prediction
        env_data = models.EnvironmentalData(
            date=day.date,
            location=location,
            rainfall_mm=day.rainfall_mm,
            vegetation_index=current_veg_index
        )
        
        # Get risk prediction
        prediction = ml_predictor.predict_single_location(db, env_data)
        
        # Enhance prediction with weather data
        enhanced_prediction = {
            **prediction,
            'date': day.date.isoformat(),
            'rainfall_mm': day.rainfall_mm,
            'temperature_c': day.temperature,
            'weather_condition': day.condition,
            'vegetation_index': current_veg_index
        }
        
        risk_forecast_list.append(enhanced_prediction)
        
        if prediction["risk_level"] == "High":
            high_risk_days += 1
    
    # Determine overall trend
    if high_risk_days >= 3:
        trend = "📈 Increasing risk"
        trend_description = "Multiple high-risk days ahead"
    elif high_risk_days >= 1:
        trend = "⚠️ Moderate risk" 
        trend_description = "Some high-risk periods expected"
    else:
        trend = "✅ Low risk"
        trend_description = "Generally low risk period"
    
    return {
        "location": location,
        "coordinates": coords if coords else {"latitude": None, "longitude": None},
        "forecast_generated_at": pd.Timestamp.now().isoformat(),
        "data_source": "WeatherAPI.com + Historical Analysis",
        "vegetation_index_used": current_veg_index,
        "risk_forecast": risk_forecast_list,
        "summary": {
            "total_days": len(risk_forecast_list),
            "high_risk_days": high_risk_days,
            "trend": trend,
            "trend_description": trend_description,
            "recommendation": "Increase patrols" if high_risk_days >= 2 else "Normal monitoring"
        }
    }
# --- Alert Management Endpoints ---

@app.get("/alerts/", response_model=List[schemas.Alert], tags=["Alerts"])
def get_alerts(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Get all sent alerts"""
    alerts = db.query(models.Alert).offset(skip).limit(limit).all()
    return alerts

@app.post("/alerts/test", tags=["Alerts"])
def test_alert_system(location: str = "Hambantota"):
    """Test the alert system (for demonstration)"""
    print(f"🚨 TEST: Triggering test alert for {location}")
    dispatch_alerts(location)
    
    return {
        "message": f"Test alert triggered for {location}",
        "alert_type": "SMS & Email",
        "test_mode": True,
        "timestamp": pd.Timestamp.now().isoformat()
    }

# --- Advanced Analytics Endpoints ---

@app.get("/advanced-analytics/territories", tags=["Advanced Analytics"])
def get_territory_analysis(db: Session = Depends(get_db)):
    """Advanced territory analysis (placeholder for herd analysis)"""
    # This would integrate with your HerdTerritoryAnalyzer
    return {
        "feature": "territory_analysis",
        "status": "development",
        "description": "Elephant herd territory mapping and analysis",
        "available_soon": True
    }

@app.get("/advanced-analytics/economic-impact", tags=["Advanced Analytics"])
def get_economic_impact_analysis(db: Session = Depends(get_db)):
    """Economic impact analysis (placeholder)"""
    return {
        "feature": "economic_impact",
        "status": "development", 
        "description": "Economic risk quantification and ROI analysis",
        "available_soon": True
    }

# --- Error Handling ---

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "advice": "Check server logs and ensure database is properly initialized",
            "support": "Run python run_day1.py to reset database if needed"
        }
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Resource not found",
            "available_endpoints": [
                "/", "/health", "/incidents/", "/environmental-data/", 
                "/train-model/", "/predict-risk/", "/risk-heatmap/", "/analytics/",
                "/report-sighting/", "/predict-forecast/{location}", "/system-status"
            ]
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Elephant Conflict Early Warning System API...")
    print("🔓 Security temporarily disabled for demo purposes")
    print("📚 API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)