# File: app/severity_predictor.py

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np

class ConflictSeverityPredictor:
    def __init__(self):
        self.damage_model = None
        self.severity_model = None
        
    def prepare_training_data(self, conflicts_df):
        """Engineer features for severity prediction"""
        
        # Create economic impact labels (you'll need to estimate these)
        # Based on conflict type
        damage_mapping = {
            'crop_raiding': 50000,      # Rs. 50K average
            'property_damage': 120000,   # Rs. 120K average
            'human_injury': 200000,      # Rs. 200K average (medical + lost work)
            'water_source': 20000        # Rs. 20K average
        }
        
        conflicts_df['estimated_damage'] = conflicts_df['conflict_type'].map(damage_mapping)
        
        # Add multiplier based on severity (if you have this field)
        # If not, create synthetic severity based on other factors
        
        # Severity classification
        def classify_severity(damage):
            if damage < 30000:
                return 0  # Low
            elif damage < 100000:
                return 1  # Medium
            elif damage < 180000:
                return 2  # High
            else:
                return 3  # Critical
        
        conflicts_df['severity_class'] = conflicts_df['estimated_damage'].apply(classify_severity)
        
        # Features
        features = ['temperature', 'rainfall', 'humidity', 'crop_season', 
                    'distance_to_forest', 'previous_incidents_30days']
        
        X = conflicts_df[features]
        y_damage = conflicts_df['estimated_damage']
        y_severity = conflicts_df['severity_class']
        
        return X, y_damage, y_severity
    
    def train_multi_output_model(self, X_train, y_damage_train, y_severity_train):
        """Train models for damage and severity prediction"""
        
        # Model 1: Damage cost prediction (regression)
        self.damage_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.damage_model.fit(X_train, y_damage_train)
        
        # Model 2: Severity classification
        self.severity_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.severity_model.fit(X_train, y_severity_train)
        
        return self
    
    def predict_severity(self, X_new):
        """Predict both damage and severity for new data"""
        
        damage_pred = self.damage_model.predict(X_new)
        severity_pred = self.severity_model.predict(X_new)
        severity_proba = self.severity_model.predict_proba(X_new)
        
        severity_labels = ['Low', 'Medium', 'High', 'Critical']
        
        results = []
        for i in range(len(X_new)):
            results.append({
                'estimated_damage_rs': int(damage_pred[i]),
                'severity_level': severity_labels[severity_pred[i]],
                'confidence': max(severity_proba[i])
            })
        
        return results
    
    def prioritize_response_resources(self, predictions_df):
        """Rank locations by risk × potential damage"""
        
        # Calculate priority score
        predictions_df['priority_score'] = (
            predictions_df['risk_probability'] / 100 *  # 0-1 scale
            predictions_df['estimated_damage_rs'] / 1000  # Scaled damage
        )
        
        # Rank locations
        predictions_df['priority_rank'] = predictions_df['priority_score'].rank(ascending=False)
        
        # Categorize response urgency
        def get_response_level(score):
            if score > 150:
                return 'IMMEDIATE - Deploy 3+ patrol units'
            elif score > 80:
                return 'URGENT - Deploy 2 patrol units'
            elif score > 40:
                return 'MODERATE - Deploy 1 patrol unit'
            else:
                return 'MONITOR - Remote surveillance'
        
        predictions_df['response_recommendation'] = predictions_df['priority_score'].apply(get_response_level)
        
        return predictions_df.sort_values('priority_rank')