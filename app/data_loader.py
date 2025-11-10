import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app import models
import os

def load_sri_lanka_conflict_data(db: Session):
    """Load or create Sri Lanka conflict data"""
    
    # Try to load from CSV if exists, else create synthetic data
    csv_path = "app/data/sri_lanka_elephant_conflict.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} real incidents from CSV")
    else:
        print("Creating synthetic Sri Lanka conflict data...")
        df = create_synthetic_sri_lanka_data()
        # Create directory if it doesn't exist
        os.makedirs("app/data", exist_ok=True)
        df.to_csv(csv_path, index=False)
    
    # Load into database
    for _, row in df.iterrows():
        incident = models.ConflictIncident(
            timestamp=datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S'),
            latitude=row['latitude'],
            longitude=row['longitude'],
            incident_type=row['incident_type'],
            elephant_count=row['elephant_count'],
            crop_damage_hectares=row['crop_damage_hectares'],
            village_name=row['village_name'],
            district=row['district'],
            province=row['province'],
            reported_by=row['reported_by'],
            description=row['description']
        )
        db.add(incident)
    
    db.commit()
    print(f"Loaded {len(df)} conflict incidents into database")

def create_synthetic_sri_lanka_data():
    """Create realistic Sri Lanka elephant conflict data"""
    
    # Sri Lanka districts with high human-elephant conflict
    districts = {
        'Hambantota': {'lat_range': (6.0, 6.4), 'lon_range': (81.0, 81.5)},
        'Monaragala': {'lat_range': (6.5, 7.0), 'lon_range': (81.0, 81.3)},
        'Ampara': {'lat_range': (7.0, 7.5), 'lon_range': (81.3, 81.8)},
        'Polonnaruwa': {'lat_range': (7.8, 8.2), 'lon_range': (80.8, 81.2)},
        'Anuradhapura': {'lat_range': (8.1, 8.6), 'lon_range': (80.3, 80.8)}
    }
    
    villages = {
        'Hambantota': ['Thanamalwila', 'Weerawila', 'Tissamaharama', 'Deberawewa'],
        'Monaragala': ['Buttala', 'Wellawaya', 'Katharagama', 'Medagana'],
        'Ampara': ['Uhana', 'Padiyatalawa', 'Mahaoya', 'Damana'],
        'Polonnaruwa': ['Hingurakgoda', 'Medirigiriya', 'Welikanda', 'Dimbulagala'],
        'Anuradhapura': ['Kebithigollewa', 'Horowpothana', 'Mihintale', 'Nochchiyagama']
    }
    
    incident_types = ['crop_raid', 'property_damage', 'human_injury', 'elephant_death']
    
    data = []
    
    # Generate 2 years of data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    current_date = start_date
    incident_id = 1
    
    while current_date <= end_date:
        # More incidents during dry season (Jan-Apr, Jul-Sep)
        month = current_date.month
        if month in [1, 2, 3, 4, 7, 8, 9]:
            daily_incidents = np.random.poisson(1.5)  # Higher during dry season
        else:
            daily_incidents = np.random.poisson(0.7)  # Lower during wet season
        
        for _ in range(daily_incidents):
            district = np.random.choice(list(districts.keys()))
            lat_range = districts[district]['lat_range']
            lon_range = districts[district]['lon_range']
            
            lat = np.random.uniform(lat_range[0], lat_range[1])
            lon = np.random.uniform(lon_range[0], lon_range[1])
            
            incident_type = np.random.choice(incident_types, p=[0.6, 0.25, 0.1, 0.05])
            
            if incident_type == 'crop_raid':
                elephant_count = np.random.randint(1, 8)
                crop_damage = np.random.uniform(0.1, 5.0)
            else:
                elephant_count = np.random.randint(1, 4)
                crop_damage = 0.0
            
            village = np.random.choice(villages[district])
            
            data.append({
                'timestamp': (current_date + timedelta(hours=np.random.randint(0, 24))).strftime('%Y-%m-%d %H:%M:%S'),
                'latitude': round(lat, 4),
                'longitude': round(lon, 4),
                'incident_type': incident_type,
                'elephant_count': elephant_count,
                'crop_damage_hectares': round(crop_damage, 2),
                'village_name': village,
                'district': district,
                'province': 'Uva' if district == 'Monaragala' else 'Eastern' if district == 'Ampara' else 'North Central' if district in ['Polonnaruwa', 'Anuradhapura'] else 'Southern',
                'reported_by': f'Farmer_{np.random.randint(1, 100)}',
                'description': f'{incident_type.replace("_", " ").title()} reported in {village}'
            })
            
            incident_id += 1
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)

def load_environmental_data(db: Session):
    """Create synthetic environmental data"""
    print("Creating synthetic environmental data...")
    
    # Sri Lanka coordinates
    locations = [
        (6.3, 81.2), (6.7, 81.1), (7.2, 81.5), (8.0, 80.9), (8.3, 80.5),
        (6.5, 80.9), (7.5, 81.0), (8.5, 80.6), (6.8, 81.4), (7.8, 81.1)
    ]
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    current_date = start_date
    data_id = 1
    
    while current_date <= end_date:
        month = current_date.month
        
        for lat, lon in locations:
            # Seasonal patterns
            if month in [5, 6, 10, 11]:  # Monsoon seasons
                rainfall = np.random.uniform(100, 300)
                ndvi = np.random.uniform(0.6, 0.9)  # High vegetation
            else:  # Dry seasons
                rainfall = np.random.uniform(10, 80)
                ndvi = np.random.uniform(0.3, 0.6)  # Low vegetation
            
            temperature = np.random.uniform(25, 32)
            soil_moisture = np.random.uniform(0.2, 0.8)
            drought_index = np.random.uniform(0.1, 0.9)
            
            env_data = models.EnvironmentalData(
                date=current_date,
                latitude=lat,
                longitude=lon,
                rainfall_mm=rainfall,
                temperature_c=temperature,
                ndvi_vegetation_index=ndvi,
                soil_moisture=soil_moisture,
                drought_index=drought_index
            )
            db.add(env_data)
            data_id += 1
        
        current_date += timedelta(days=30)  # Monthly data
    
    db.commit()
    print("Environmental data loaded into database")

def initialize_database():
    """Initialize database with all tables and sample data"""
    from app.database import engine, SessionLocal
    
    # Create all tables
    models.Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_incidents = db.query(models.ConflictIncident).count()
        
        if existing_incidents == 0:
            print("Loading initial data...")
            load_sri_lanka_conflict_data(db)
            load_environmental_data(db)
            print("Database initialization complete!")
        else:
            print(f"Database already contains {existing_incidents} incidents")
            
    finally:
        db.close()