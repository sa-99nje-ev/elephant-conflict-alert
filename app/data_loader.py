import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.database import Base, engine, get_db  # <-- THE FIX IS HERE
from app.models import ConflictIncident, EnvironmentalData

# Define locations (matches ml_predictor and dashboard)
LOCATIONS = [
    "Anuradhapura", "Polonnaruwa", "Ampara",
    "Monaragala", "Puttalam", "Hambantota"
]
INCIDENT_TYPES = ["crop_raid", "property_damage", "sighting", "human_injury"]

def create_synthetic_data(db: Session, num_incidents: int = 400, num_env_days: int = 365):
    print("Creating synthetic data...")
    
    # --- 1. Create Synthetic Incidents ---
    incidents = []
    for _ in range(num_incidents):
        rand_days_ago = random.randint(1, num_env_days)
        incident = ConflictIncident(
            timestamp=datetime.now() - timedelta(days=rand_days_ago),
            location=random.choice(LOCATIONS),
            incident_type=random.choice(INCIDENT_TYPES),
            description="Synthetic incident report."
        )
        incidents.append(incident)
    db.add_all(incidents)
    
    # --- 2. Create Synthetic Environmental Data ---
    env_data = []
    start_date = datetime.now().date() - timedelta(days=num_env_days)
    
    for location in LOCATIONS:
        for i in range(num_env_days):
            current_date = start_date + timedelta(days=i)
            
            # Simulate seasonal rainfall
            month = current_date.month
            if 3 <= month <= 5 or 10 <= month <= 12: # Rainy seasons
                rainfall = random.uniform(5.0, 50.0)
                veg_index = random.uniform(0.6, 0.9)
            else: # Dry seasons
                rainfall = random.uniform(0.0, 10.0)
                veg_index = random.uniform(0.2, 0.5)
                
            env = EnvironmentalData(
                date=current_date,
                location=location,
                rainfall_mm=round(rainfall, 2),
                vegetation_index=round(veg_index, 2)
            )
            env_data.append(env)
            
    db.add_all(env_data)
    
    try:
        db.commit()
        print(f"Successfully added {len(incidents)} incidents and {len(env_data)} env records.")
    except Exception as e:
        print(f"Error adding synthetic data: {e}")
        db.rollback()

def initialize_database():
    print("Initializing database...")
    
    # Drop all tables first
    Base.metadata.drop_all(bind=engine)
    print("Dropped old tables.")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Created new tables.")
    
    # Get a new session
    db = next(get_db())  # <-- THE FIX IS HERE
    
    try:
        # Check if data exists
        incident_count = db.query(ConflictIncident).count()
        if incident_count == 0:
            print("No data found. Populating database...")
            create_synthetic_data(db)
        else:
            print(f"Database already populated with {incident_count} incidents.")
    finally:
        db.close()
    
    print("\n--- Database Initialization Complete ---")
    
    # Show recent data
    db = next(get_db())  # <-- THE FIX IS HERE
    try:
        print("Recent incidents:")
        recent_incidents = db.query(ConflictIncident).order_by(ConflictIncident.timestamp.desc()).limit(3).all()
        for inc in recent_incidents:
            print(f"  - {inc.timestamp.date()}: {inc.incident_type} in {inc.location}")
    finally:
        db.close()