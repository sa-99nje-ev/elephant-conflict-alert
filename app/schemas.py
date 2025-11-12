# app/schemas.py

from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date

# --- Base Schemas ---

class ConflictIncidentBase(BaseModel):
    location: str
    incident_type: str
    description: Optional[str] = None

class EnvironmentalDataBase(BaseModel):
    location: str
    date: date
    rainfall_mm: float
    vegetation_index: float

class RiskForecastBase(BaseModel):
    location: str
    forecasted_at: datetime
    risk_level: str

class AlertBase(BaseModel):
    location: str
    risk_level: str
    message: str

class FarmerReportBase(BaseModel):
    location: str
    elephant_count: int = 1
    description: Optional[str] = None

# --- Create Schemas (for API input) ---

class ConflictIncidentCreate(ConflictIncidentBase):
    pass

class EnvironmentalDataCreate(EnvironmentalDataBase):  # <-- THIS IS THE CLASS THAT WAS MISSING
    pass

class RiskForecastCreate(RiskForecastBase):
    pass

class AlertCreate(AlertBase):
    pass

class FarmerReportCreate(FarmerReportBase):
    pass

# --- Read Schemas (for API output) ---

class ConflictIncident(ConflictIncidentBase):
    id: int
    timestamp: datetime
    
    class Config:
        orm_mode = True

class EnvironmentalData(EnvironmentalDataBase):
    id: int
    
    class Config:
        orm_mode = True

class RiskForecast(RiskForecastBase):
    id: int
    
    class Config:
        orm_mode = True

class Alert(AlertBase):
    id: int
    sent_at: datetime
    
    class Config:
        orm_mode = True

class FarmerReport(FarmerReportBase):
    id: int
    reported_at: datetime
    is_verified: bool
    
    class Config:
        orm_mode = True