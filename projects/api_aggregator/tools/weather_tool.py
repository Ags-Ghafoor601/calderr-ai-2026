"""
Weather Tool — Fetches real-time weather data from wttr.in
============================================================
wttr.in is a free, no-API-key-required weather service.
It returns data in JSON format when queried with ?format=j1.
"""

import httpx
import json
from datetime import datetime


async def fetch_weather(city: str = "New York") -> dict:
    """
    Fetch current weather data for a city using wttr.in (no API key needed).

    Args:
        city: City name (e.g., "London", "Tokyo", "New York")

    Returns:
        Dictionary with weather data including temperature, condition,
        humidity, wind speed, and feels-like temperature.
    """
    url = f"https://wttr.in/{city}?format=j1"
    headers = {"User-Agent": "CalderR-API-Aggregator/1.0"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        current = data.get("current_condition", [{}])[0]
        location = data.get("nearest_area", [{}])[0]

        # Extract location info
        area_name = location.get("areaName", [{}])[0].get("value", city)
        country = location.get("country", [{}])[0].get("value", "Unknown")
        region = location.get("region", [{}])[0].get("value", "")

        # Extract current conditions
        result = {
            "city": area_name,
            "country": country,
            "region": region,
            "temperature_f": current.get("temp_F", "N/A"),
            "temperature_c": current.get("temp_C", "N/A"),
            "feels_like_f": current.get("FeelsLikeF", "N/A"),
            "feels_like_c": current.get("FeelsLikeC", "N/A"),
            "condition": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
            "humidity": current.get("humidity", "N/A"),
            "wind_speed_mph": current.get("windspeedMiles", "N/A"),
            "wind_direction": current.get("winddir16Point", "N/A"),
            "visibility_miles": current.get("visibility", "N/A"),
            "uv_index": current.get("uvIndex", "N/A"),
            "pressure_mb": current.get("pressure", "N/A"),
            "cloud_cover": current.get("cloudcover", "N/A"),
            "observation_time": current.get("observation_time", "N/A"),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "wttr.in",
            "status": "success",
        }

        # Try to get today's forecast
        weather_list = data.get("weather", [])
        if weather_list:
            today = weather_list[0]
            result["max_temp_f"] = today.get("maxtempF", "N/A")
            result["min_temp_f"] = today.get("mintempF", "N/A")
            result["max_temp_c"] = today.get("maxtempC", "N/A")
            result["min_temp_c"] = today.get("mintempC", "N/A")
            result["sunrise"] = today.get("astronomy", [{}])[0].get("sunrise", "N/A")
            result["sunset"] = today.get("astronomy", [{}])[0].get("sunset", "N/A")

        return result

    except httpx.TimeoutException:
        return {
            "city": city,
            "status": "error",
            "error": "Request timed out — weather service unreachable",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except httpx.HTTPStatusError as e:
        return {
            "city": city,
            "status": "error",
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {
            "city": city,
            "status": "error",
            "error": str(e),
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
