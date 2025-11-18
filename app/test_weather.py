#!/usr/bin/env python3
"""
Test WeatherAPI.com integration
"""

import asyncio
import os
from dotenv import load_dotenv
from app.weather_fetcher import get_weather_forecast

async def test_all_locations():
    """Test weather API with all locations"""
    load_dotenv()
    
    locations = ["Hambantota", "Monaragala", "Ampara", "Polonnaruwa", "Anuradhapura"]
    
    print("🧪 Testing WeatherAPI.com with all locations...")
    print(f"🔑 API Key: {'✅ Loaded' if os.getenv('WEATHERAPI_KEY') else '❌ Missing'}")
    
    for location in locations:
        print(f"\n📍 Testing {location}...")
        try:
            forecast = await get_weather_forecast(location)
            print(f"   ✅ Success: {len(forecast)} days")
            for day in forecast[:2]:  # Show first 2 days
                print(f"      {day.date}: {day.rainfall_mm}mm, {day.temperature}°C, {day.condition}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_locations())