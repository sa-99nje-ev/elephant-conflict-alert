# File: app/terrain_analyzer.py

import elevation
import rasterio
from scipy.ndimage import gaussian_filter
import numpy as np

class TerrainRiskModel:
    def __init__(self):
        self.elevation_data = None
        
    def download_elevation_data(self, bounds):
        """Download SRTM elevation data"""
        # bounds = (min_lon, min_lat, max_lon, max_lat)
        
        output_file = 'data/sri_lanka_elevation.tif'
        elevation.clip(bounds=bounds, output=output_file)
        
        # Load elevation raster
        with rasterio.open(output_file) as src:
            self.elevation_data = src.read(1)
            self.transform = src.transform
            
        return self
    
    def calculate_slope(self):
        """Calculate slope from elevation"""
        
        # Calculate gradient
        dy, dx = np.gradient(self.elevation_data)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        return slope
    
    def identify_valleys(self, slope_data):
        """Find valleys (low slope areas) - elephant corridors"""
        
        # Smooth data
        slope_smooth = gaussian_filter(slope_data, sigma=2)
        
        # Valleys = slope < 15 degrees
        valleys = slope_smooth < 15
        
        return valleys
    
    def calculate_terrain_risk(self, location_lat, location_lon):
        """Calculate risk based on terrain features"""
        
        # Convert lat/lon to pixel coordinates
        row, col = ~self.transform * (location_lon, location_lat)
        row, col = int(row), int(col)
        
        # Get local elevation and slope
        elevation = self.elevation_data[row, col]
        slope = self.calculate_slope()[row, col]
        
        # Risk factors
        is_valley = slope < 15  # Elephants prefer valleys
        is_near_water = self.check_water_proximity(location_lat, location_lon)
        
        risk_score = 0
        if is_valley:
            risk_score += 30
        if elevation < 500:  # Low elevation
            risk_score += 20
        if is_near_water:
            risk_score += 40
            
        return risk_score
    
    def check_water_proximity(self, lat, lon, radius_km=2):
        """Check if location is near water sources"""
        
        # Use Overpass API to get water sources
        import requests
        
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        (
          node["natural"="water"](around:{radius_km*1000},{lat},{lon});
          way["natural"="water"](around:{radius_km*1000},{lat},{lon});
        );
        out center;
        """
        
        try:
            response = requests.get(overpass_url, params={'data': query}, timeout=30)
            data = response.json()
            
            return len(data['elements']) > 0
        except:
            return False  # If API fails, return False

# In your ML predictor, add terrain features
def add_terrain_features(df, terrain_model):
    """Add terrain risk scores to your dataframe"""
    
    df['terrain_risk_score'] = df.apply(
        lambda row: terrain_model.calculate_terrain_risk(row['latitude'], row['longitude']),
        axis=1
    )
    
    return df