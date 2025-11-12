import os
import httpx
from dotenv import load_dotenv
from datetime import datetime, date
from collections import defaultdict

# Load API key from .env file
load_dotenv()
OWM_API_KEY = os.getenv("OWM_API_KEY")

# --- THE FIX IS HERE ---
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

class ForecastDay:
    """A simple class to hold our forecast data."""
    def __init__(self, dt, rainfall_mm, temp_c):
        self.date = date.fromtimestamp(dt)
        self.rainfall_mm = rainfall_mm
        self.temperature = temp_c

    def __repr__(self):
        return f"<ForecastDay(date={self.date}, rain={self.rainfall_mm}mm)>"

async def get_weather_forecast(lat: float, lon: float) -> list[ForecastDay]:
    """
    Fetches the 5-day weather forecast for a given lat/lon.
    """
    
    if not OWM_API_KEY:
        print("--- !!! DEBUG: OWM_API_KEY not found in .env file. ---")
        return []
    else:
        print(f"--- DEBUG: Found OWM_API_KEY ending in: ...{OWM_API_KEY[-4:]}")

    params = {
        "lat": lat,
        "lon": lon,
        "appid": OWM_API_KEY,
        "units": "metric"
    }

    try:
        print(f"--- DEBUG: Calling OpenWeatherMap API at: {BASE_URL} ---")
        async with httpx.AsyncClient() as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status() # Raise an error for bad responses
        
        print("--- DEBUG: API call successful, processing data... ---")
        data = response.json()
        
        daily_data = defaultdict(lambda: {"rainfall": 0, "temps": [], "dt": None})
        
        for item in data.get("list", []):
            day_key = date.fromtimestamp(item.get("dt")).isoformat()
            daily_data[day_key]["rainfall"] += item.get("rain", {}).get("3h", 0)
            daily_data[day_key]["temps"].append(item.get("main", {}).get("temp", 25))
            if not daily_data[day_key]["dt"]:
                daily_data[day_key]["dt"] = item.get("dt")
        
        forecast_list = []
        for day in daily_data.values():
            if not day["dt"]:
                continue
            avg_temp = sum(day["temps"]) / len(day["temps"])
            forecast_list.append(
                ForecastDay(
                    dt=day["dt"],
                    rainfall_mm=day["rainfall"],
                    temp_c=avg_temp
                )
            )
        
        print(f"--- DEBUG: Processed {len(forecast_list)} forecast days. ---")
        return forecast_list[:5]
        
    except httpx.HTTPStatusError as e:
        print("\n--- !!! DEBUG: HTTP ERROR FROM OPENWEATHERMAP !!! ---")
        print(f"Error fetching weather data: {e.response.status_code}")
        print(f"Response: {e.response.json()}")
        print("--- !!! END DEBUG !!! ---\n")
        return []
    except Exception as e:
        print(f"\n--- !!! DEBUG: AN UNEXPECTED ERROR OCCURRED !!! ---")
        print(f"Error: {e}")
        print("--- !!! END DEBUG !!! ---\n")
        return []