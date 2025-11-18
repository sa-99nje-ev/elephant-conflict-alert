from app.terrain_analyzer import TerrainRiskModel

# Sri Lanka bounding box (min_lon, min_lat, max_lon, max_lat)
bounds = (79.5, 5.5, 82.0, 10.0)

print("Downloading SRTM elevation data...")

model = TerrainRiskModel()
model.download_elevation_data(bounds)

print("Elevation data saved to data/sri_lanka_elevation.tif")
