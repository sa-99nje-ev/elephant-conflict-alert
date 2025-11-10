from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class ConflictIncidentBase(BaseModel):
    timestamp: datetime
    latitude: float
    longitude: float
    incident_type: str
    elephant_count: Optional[int] = None
    crop_damage_hectares: Optional[float] = None
    village_name: Optional[str] = None
    district: Optional[str] = None
    province: Optional[str] = None
    reported_by: Optional[str] = None
    description: Optional[str] = None

class ConflictIncidentCreate(ConflictIncidentBase):
    pass

class ConflictIncident(ConflictIncidentBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class EnvironmentalDataBase(BaseModel):
    date: datetime
    latitude: float
    longitude: float
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None
    ndvi_vegetation_index: Optional[float] = None
    soil_moisture: Optional[float] = None
    drought_index: Optional[float] = None

class EnvironmentalData(EnvironmentalDataBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class RiskForecastBase(BaseModel):
    forecast_date: datetime
    latitude: float
    longitude: float
    risk_score: float
    risk_level: str
    confidence: Optional[float] = None
    factors: Optional[str] = None
    district: Optional[str] = None

class RiskForecast(RiskForecastBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class AlertBase(BaseModel):
    alert_date: datetime
    latitude: float
    longitude: float
    alert_type: str
    message: str
    district: Optional[str] = None
    village_name: Optional[str] = None

class Alert(AlertBase):
    id: int
    is_sent: bool
    sent_via: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Response models for API
class RiskHeatmapResponse(BaseModel):
    latitude: float
    longitude: float
    risk_score: float
    risk_level: str
    district: str

class StatsResponse(BaseModel):
    total_incidents: int
    high_risk_zones: int
    alerts_sent: int
    prediction_accuracy: Optional[float] = None

class DistrictSummary(BaseModel):
    name: str
    incident_count: int
    last_incident: Optional[datetime] = None