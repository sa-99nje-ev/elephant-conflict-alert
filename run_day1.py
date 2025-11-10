#!/usr/bin/env python3
"""
Day 1: Initialize the Elephant Conflict Database
"""

from app.data_loader import initialize_database
from app.database import SessionLocal
from app import models

def main():
    print("🚀 ELEPHANT CONFLICT EARLY WARNING SYSTEM - DAY 1")
    print("=" * 50)
    
    # Initialize database
    initialize_database()
    
    # Show some stats
    db = SessionLocal()
    try:
        incident_count = db.query(models.ConflictIncident).count()
        env_count = db.query(models.EnvironmentalData).count()
        
        print(f"✅ Database initialized successfully!")
        print(f"📊 Loaded {incident_count} conflict incidents")
        print(f"🌧️  Loaded {env_count} environmental data points")
        print(f"🗃️  Database file: elephant_conflict.db")
        
        # Show districts with incident counts
        districts = db.query(models.ConflictIncident.district).distinct().all()
        print(f"📍 Districts covered: {[d[0] for d in districts]}")
        
        # Show recent incidents
        recent = db.query(models.ConflictIncident).order_by(models.ConflictIncident.timestamp.desc()).limit(3).all()
        print(f"\n📅 Recent incidents:")
        for incident in recent:
            print(f"   - {incident.timestamp.date()}: {incident.incident_type} in {incident.district}")
        
    finally:
        db.close()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run: uvicorn main:app --reload")
    print("2. Visit: http://localhost:8000/docs for API documentation")
    print("3. Check database with: sqlite3 elephant_conflict.db")

if __name__ == "__main__":
    main()