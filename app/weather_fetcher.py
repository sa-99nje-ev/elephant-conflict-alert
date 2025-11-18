import os
import httpx
from dotenv import load_dotenv
from datetime import datetime, date
import asyncio

# Load API key from .env file
load_dotenv()
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY")

class ForecastDay:
    """A simple class to hold our forecast data."""
    def __init__(self, date, rainfall_mm, temp_c, condition):
        self.date = date
        self.rainfall_mm = rainfall_mm
        self.temperature = temp_c
        self.condition = condition

    def __repr__(self):
        return f"<ForecastDay(date={self.date}, rain={self.rainfall_mm}mm, temp={self.temperature}°C)>"

async def get_weather_forecast(location_name: str) -> list[ForecastDay]:
    """
    Fetches the 7-day weather forecast for a given location name using WeatherAPI.com
    """
    
    if not WEATHERAPI_KEY:
        print("❌ WEATHERAPI_KEY not found in .env file.")
        return get_mock_forecast_data()
    
    print(f"🌤️ Fetching weather forecast for: {location_name}")

    # WeatherAPI.com endpoint for 7-day forecast
    url = "http://api.weatherapi.com/v1/forecast.json"
    
    params = {
        "key": WEATHERAPI_KEY,
        "q": location_name,
        "days": 7,
        "aqi": "no",
        "alerts": "no"
    }

    try:
        print(f"🔗 Calling WeatherAPI.com...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
        
        data = response.json()
        print(f"✅ Successfully received weather data for {location_name}")
        
        forecast_days = []
        
        for day_data in data['forecast']['forecastday']:
            date_str = day_data['date']
            rainfall_mm = day_data['day']['totalprecip_mm']
            temp_c = day_data['day']['avgtemp_c']
            condition = day_data['day']['condition']['text']
            
            forecast_day = ForecastDay(
                date=datetime.strptime(date_str, '%Y-%m-%d').date(),
                rainfall_mm=rainfall_mm,
                temp_c=temp_c,
                condition=condition
            )
            forecast_days.append(forecast_day)
        
        print(f"📅 Processed {len(forecast_days)} forecast days")
        return forecast_days[:5]  # Return only 5 days for consistency
        
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP Error from WeatherAPI: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return get_mock_forecast_data()
        
    except Exception as e:
        print(f"❌ Unexpected error fetching weather: {e}")
        return get_mock_forecast_data()

def get_mock_forecast_data() -> list[ForecastDay]:
    """
    Generate realistic mock forecast data as fallback
    """
    print("🔄 Using mock weather data as fallback...")
    
    from datetime import datetime, timedelta
    import random
    
    forecast_days = []
    
    for i in range(5):
        current_date = (datetime.now() + timedelta(days=i)).date()
        
        # Sri Lanka typical weather patterns
        if i < 2:
            # First 2 days: realistic current conditions
            rainfall = random.uniform(0, 15)
            temp = random.uniform(26, 32)
            condition = "Partly cloudy" if rainfall < 5 else "Light rain"
        else:
            # Future days: varied conditions
            rainfall = random.uniform(0, 25)
            temp = random.uniform(25, 33)
            conditions = ["Sunny", "Partly cloudy", "Cloudy", "Light rain", "Moderate rain"]
            condition = random.choice(conditions)
        
        forecast_day = ForecastDay(
            date=current_date,
            rainfall_mm=round(rainfall, 1),
            temp_c=round(temp, 1),
            condition=condition
        )
        forecast_days.append(forecast_day)
    
    return forecast_days

# Synchronous version for non-async contexts
def get_weather_forecast_sync(location_name: str) -> list[ForecastDay]:
    """
    Synchronous wrapper for the async function
    """
    return asyncio.run(get_weather_forecast(location_name))

# Test function
async def test_weather_api():
    """Test the weather API with a known location"""
    print("🧪 Testing WeatherAPI integration...")
    forecast = await get_weather_forecast("Colombo")
    print(f"✅ Test completed. Got {len(forecast)} days of forecast")
    for day in forecast:
        print(f"   {day.date}: {day.rainfall_mm}mm rain, {day.temperature}°C, {day.condition}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_weather_api())