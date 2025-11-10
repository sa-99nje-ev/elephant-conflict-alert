from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import json

from app.database import get_db, engine
from app import models, schemas
from app.data_loader import initialize_database

# Create FastAPI app
app = FastAPI(
    title="Elephant Conflict Early Warning System",
    description="Predictive analytics for human-elephant conflict in Sri Lanka",
    version="1.0.0"
)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    initialize_database()

# Basic health check
@app.get("/")
def read_root():
    return {
        "message": "Elephant Conflict Early Warning System API",
        "status": "active",
        "version": "1.0.0"
    }

# Conflict incidents endpoints
@app.get("/incidents/", response_model=List[schemas.ConflictIncident])
def get_incidents(
    skip: int = 0,
    limit: int = 100,
    district: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get conflict incidents with optional filters"""
    query = db.query(models.ConflictIncident)
    
    if district:
        query = query.filter(models.ConflictIncident.district == district)
    
    if start_date:
        query = query.filter(models.ConflictIncident.timestamp >= start_date)
    
    if end_date:
        query = query.filter(models.ConflictIncident.timestamp <= end_date)
    
    return query.order_by(models.ConflictIncident.timestamp.desc()).offset(skip).limit(limit).all()

@app.post("/incidents/", response_model=schemas.ConflictIncident)
def create_incident(incident: schemas.ConflictIncidentCreate, db: Session = Depends(get_db)):
    """Create a new conflict incident report"""
    db_incident = models.ConflictIncident(**incident.dict())
    db.add(db_incident)
    db.commit()
    db.refresh(db_incident)
    return db_incident

# Statistics endpoint
@app.get("/stats/", response_model=schemas.StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    total_incidents = db.query(models.ConflictIncident).count()
    
    # High risk zones (incidents in last 3 months)
    three_months_ago = datetime.now() - timedelta(days=90)
    recent_incidents = db.query(models.ConflictIncident).filter(
        models.ConflictIncident.timestamp >= three_months_ago
    ).count()
    
    alerts_sent = db.query(models.Alert).filter(models.Alert.is_sent == True).count()
    
    return schemas.StatsResponse(
        total_incidents=total_incidents,
        high_risk_zones=recent_incidents,
        alerts_sent=alerts_sent,
        prediction_accuracy=0.85  # Placeholder for now
    )

# District list endpoint
@app.get("/districts/")
def get_districts(db: Session = Depends(get_db)):
    """Get unique districts with incident counts"""
    districts = db.query(
        models.ConflictIncident.district
    ).filter(models.ConflictIncident.district.isnot(None)).distinct().all()
    
    district_list = []
    for district in districts:
        count = db.query(models.ConflictIncident).filter(
            models.ConflictIncident.district == district[0]
        ).count()
        district_list.append({"name": district[0], "incident_count": count})
    
    return district_list

# Environmental data endpoints
@app.get("/environment/", response_model=List[schemas.EnvironmentalData])
def get_environmental_data(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get environmental data"""
    return db.query(models.EnvironmentalData).offset(skip).limit(limit).all()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)