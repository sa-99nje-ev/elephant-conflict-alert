# File: app/herd_analyzer.py

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from datetime import datetime

class HerdTerritoryAnalyzer:
    def __init__(self, db_session):
        self.db = db_session
        
    def cluster_conflict_zones(self):
        """Identify distinct elephant territories using DBSCAN"""
        # Get all conflicts with coordinates
        conflicts = pd.read_sql("SELECT * FROM conflicts", self.db)
        
        # Prepare coordinates (lat, lon in radians for earth distance)
        coords = np.radians(conflicts[['latitude', 'longitude']].values)
        
        # DBSCAN clustering
        # eps=0.05 radians ≈ 5.5 km (tune based on your data)
        # min_samples=3 (at least 3 conflicts to form a territory)
        clustering = DBSCAN(eps=0.05, min_samples=3, metric='haversine')
        conflicts['territory_id'] = clustering.fit_predict(coords)
        
        # Remove noise points (territory_id = -1)
        territories = conflicts[conflicts['territory_id'] != -1]
        
        # Calculate territory stats
        territory_stats = territories.groupby('territory_id').agg({
            'latitude': ['mean', 'std'],
            'longitude': ['mean', 'std'],
            'id': 'count'  # Number of conflicts
        }).reset_index()
        
        territory_stats.columns = ['territory_id', 'center_lat', 'lat_spread', 
                                    'center_lon', 'lon_spread', 'conflict_count']
        
        return territories, territory_stats
    
    def analyze_temporal_patterns(self, territories_df):
        """Find when each territory is most active"""
        territories_df['month'] = pd.to_datetime(territories_df['date']).dt.month
        territories_df['season'] = territories_df['month'].apply(self._get_season)
        
        # Activity by territory and season
        temporal_patterns = territories_df.groupby(['territory_id', 'season']).size().reset_index(name='conflicts')
        
        # Find peak season for each territory
        peak_seasons = temporal_patterns.loc[temporal_patterns.groupby('territory_id')['conflicts'].idxmax()]
        
        return temporal_patterns, peak_seasons
    
    def _get_season(self, month):
        """Sri Lanka seasons"""
        if month in [5, 6, 7, 8, 9]:
            return 'Southwest Monsoon'
        elif month in [10, 11, 12, 1, 2]:
            return 'Northeast Monsoon'
        else:
            return 'Inter-monsoon'
    
    def predict_territory_expansion(self, territories_df):
        """Check if territories are shifting over time"""
        territories_df['year'] = pd.to_datetime(territories_df['date']).dt.year
        
        expansion_data = []
        for territory_id in territories_df['territory_id'].unique():
            territory_data = territories_df[territories_df['territory_id'] == territory_id]
            
            # Calculate centroid shift year-over-year
            yearly_centroids = territory_data.groupby('year').agg({
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            
            if len(yearly_centroids) > 1:
                # Calculate distance moved (simple lat/lon difference)
                lat_shift = yearly_centroids['latitude'].iloc[-1] - yearly_centroids['latitude'].iloc[0]
                lon_shift = yearly_centroids['longitude'].iloc[-1] - yearly_centroids['longitude'].iloc[0]
                
                # Approximate distance in km
                distance_km = np.sqrt(lat_shift**2 + lon_shift**2) * 111  # 1 degree ≈ 111 km
                
                expansion_data.append({
                    'territory_id': territory_id,
                    'years_tracked': len(yearly_centroids),
                    'distance_shifted_km': round(distance_km, 2),
                    'direction': self._get_direction(lat_shift, lon_shift)
                })
        
        return pd.DataFrame(expansion_data)
    
    def _get_direction(self, lat_shift, lon_shift):
        """Determine cardinal direction of movement"""
        if abs(lat_shift) > abs(lon_shift):
            return 'North' if lat_shift > 0 else 'South'
        else:
            return 'East' if lon_shift > 0 else 'West'