from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Date, func
from .database import Base

class ConflictIncident(Base):
    __tablename__ = "conflict_incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    # --- SIMPLIFIED ---
    # We will use 'location' as the main join key, matching EnvironmentalData
    location = Column(String, index=True, nullable=False)
    incident_type = Column(String, index=True)
    description = Column(String, nullable=True)

class EnvironmentalData(Base):
    __tablename__ = "environmental_data"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # --- THIS IS THE CRITICAL FIX ---
    location = Column(String, index=True, nullable=False)
    
    rainfall_mm = Column(Float, nullable=False)
    vegetation_index = Column(Float, nullable=False) # Simplified from 'ndvi'

class RiskForecast(Base):
    __tablename__ = "risk_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    forecasted_at = Column(DateTime(timezone=True), server_default=func.now())
    location = Column(String, index=True)
    risk_level = Column(String)

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    sent_at = Column(DateTime(timezone=True), server_default=func.now())
    location = Column(String, index=True)
    risk_level = Column(String)
    message = Column(String)

class FarmerReport(Base):
    __tablename__ = "farmer_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    reported_at = Column(DateTime(timezone=True), server_default=func.now())
    location = Column(String, index=True, nullable=False)
    elephant_count = Column(Integer, default=1)
    description = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)