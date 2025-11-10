from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.sql import func
from app.database import Base

class ConflictIncident(Base):
    __tablename__ = "conflict_incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    incident_type = Column(String(100), nullable=False)
    elephant_count = Column(Integer)
    crop_damage_hectares = Column(Float)
    village_name = Column(String(200))
    district = Column(String(100))
    province = Column(String(100))
    reported_by = Column(String(200))
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class EnvironmentalData(Base):
    __tablename__ = "environmental_data"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    rainfall_mm = Column(Float)
    temperature_c = Column(Float)
    ndvi_vegetation_index = Column(Float)
    soil_moisture = Column(Float)
    drought_index = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class RiskForecast(Base):
    __tablename__ = "risk_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_date = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    confidence = Column(Float)
    factors = Column(Text)
    district = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_date = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    alert_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    district = Column(String(100))
    village_name = Column(String(200))
    is_sent = Column(Boolean, default=False)
    sent_via = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())