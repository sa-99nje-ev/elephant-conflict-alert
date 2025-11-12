import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.orm import Session
from app import models, database
import datetime

class ElephantRiskPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = [
            'month', 'rainfall_mm', 'vegetation_index', 
            'hist_conflict_density', 'is_harvest_season'
        ]
        self.is_trained = False

    def _load_and_preprocess_data(self, db: Session):
        """Loads and preprocesses data from the database."""
        incidents = pd.read_sql(db.query(models.ConflictIncident).statement, db.bind)
        if incidents.empty:
            raise ValueError("No incident data found in database to train on.")
            
        env_data = pd.read_sql(db.query(models.EnvironmentalData).statement, db.bind)
        if env_data.empty:
            raise ValueError("No environmental data found in database to train on.")

        # --- Feature Engineering ---
        incidents['timestamp'] = pd.to_datetime(incidents['timestamp'])
        env_data['date'] = pd.to_datetime(env_data['date'])
        
        # Create a matching 'date' column in incidents
        incidents['date'] = pd.to_datetime(incidents['timestamp'].dt.date)
        incidents['risk_label'] = 1
        
        # Merge on matching date and location
        data = pd.merge(env_data, incidents, 
                        on=['date', 'location'],
                        how='left')
        
        # Continue with feature engineering
        data['risk_label'] = data['risk_label'].fillna(0).astype(int)
        data['month'] = data['date'].dt.month
        data['is_harvest_season'] = data['month'].between(8, 10).astype(int)
        
        data = data.sort_values(by=['location', 'date'])
        data['hist_conflict_density'] = data.groupby('location')['risk_label'].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
        data['hist_conflict_density'] = data['hist_conflict_density'].fillna(0)

        data = data.dropna(subset=['rainfall_mm', 'vegetation_index'])
        
        return data

    def train(self, db: Session):
        """Trains the Random Forest model."""
        try:
            data = self._load_and_preprocess_data(db)
        except ValueError as e:
            print(f"Training failed: {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            print(f"Training failed: An unexpected error occurred. {e}")
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}

        if data.empty or data[self.feature_columns].empty:
            print("Training failed: No usable data after preprocessing.")
            return {"status": "error", "message": "No data to train on."}
            
        X = data[self.feature_columns]
        y = data['risk_label']

        if len(np.unique(y)) < 2:
            print("Warning: Only one class found. Training may be inaccurate.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        
        self.is_trained = True
        print(f"Model trained with accuracy: {accuracy}")
        return {"status": "success", "accuracy": accuracy, "features_used": self.feature_columns}

    def _rule_based_predict(self, features_df):
        print("Warning: Using rule-based fallback model.")
        predictions = []
        for _, row in features_df.iterrows():
            score = 0
            if row['rainfall_mm'] < 50: score += 1
            if row['vegetation_index'] > 0.6: score += 1
            if row['hist_conflict_density'] > 0.1: score += 2
            if row['is_harvest_season'] == 1: score += 2
            
            if score >= 4:
                predictions.append("High")
            elif score >= 2:
                predictions.append("Medium")
            else:
                predictions.append("Low")
        return predictions

    def predict_single_location(self, db: Session, data: models.EnvironmentalData):
        """
        Predicts risk for a single location.
        'data' is an EnvironmentalData SQLAlchemy model object.
        """
        
        # --- THIS IS THE FIX ---
        # Manually create a dictionary from the SQLAlchemy object's attributes
        # instead of calling data.dict()
        input_data = [{
            'date': data.date,
            'location': data.location,
            'rainfall_mm': data.rainfall_mm,
            'vegetation_index': data.vegetation_index
        }]
        input_df = pd.DataFrame(input_data)
        # --- END FIX ---
        
        # --- Feature Engineering (must match training) ---
        input_df['date'] = pd.to_datetime(input_df['date'])
        input_df['month'] = input_df['date'].dt.month
        input_df['is_harvest_season'] = input_df['month'].between(8, 10).astype(int)
        
        # Get historical conflict density for this location
        query = db.query(models.ConflictIncident).filter(models.ConflictIncident.location == data.location)
        hist_incidents = pd.read_sql(query.statement, db.bind)
        
        hist_conflict_density = 0
        if not hist_incidents.empty:
            recent_date = pd.to_datetime(data.date) - pd.Timedelta(days=30)
            hist_incidents['timestamp'] = pd.to_datetime(hist_incidents['timestamp'])
            recent_incidents = hist_incidents[hist_incidents['timestamp'] > recent_date]
            hist_conflict_density = len(recent_incidents) / 30.0
            
        input_df['hist_conflict_density'] = hist_conflict_density
        
        X_predict = input_df[self.feature_columns]

        # Use model or fallback
        if self.is_trained:
            probability = self.model.predict_proba(X_predict)[0][1]
            if probability > 0.7:
                risk_level = "High"
            elif probability > 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
        else:
            risk_level = self._rule_based_predict(X_predict)[0]
        
        return {
            "location": data.location,
            "risk_level": risk_level,
            "prediction_model": "RandomForest" if self.is_trained else "RuleBased"
        }

    def predict_heatmap(self, db: Session):
        """Generates risk predictions for all known locations for the dashboard."""
        locations = pd.read_sql(db.query(models.EnvironmentalData.location).distinct().statement, db.bind)
        
        predictions = []
        for loc in locations['location']:
            # Get the *latest* environmental data for this location
            latest_env_data = db.query(models.EnvironmentalData).filter(
                models.EnvironmentalData.location == loc
            ).order_by(models.EnvironmentalData.date.desc()).first()
            
            if latest_env_data:
                # This will now work!
                prediction = self.predict_single_location(db, latest_env_data)
                
                # Add coordinates
                lat, lon = self._get_coords_for_location(loc)
                prediction['latitude'] = lat
                prediction['longitude'] = lon
                predictions.append(prediction)
                
        return predictions

    def _get_coords_for_location(self, location_name: str):
        """Helper to get coordinates for dashboard map. Uses data_loader's coords."""
        locations = {
            "Anuradhapura": (8.3114, 80.4037),
            "Polonnaruwa": (7.9403, 81.0188),
            "Ampara": (7.2947, 81.6748),
            "Monaragala": (6.8724, 81.3496),
            "Puttalam": (8.0343, 79.8430),
            "HambBantota": (6.1240, 81.1185) # Corrected typo in location name
        }
        return locations.get(location_name, (7.8731, 80.7718)) # Default to Sri Lanka center

ml_predictor = ElephantRiskPredictor()