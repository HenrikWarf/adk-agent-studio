"""
Weather tool - simple function-based tool for getting weather information.
"""


async def get_weather(location: str) -> str:
    """
    Get current weather information for a location.
    
    Args:
        location: City name or location (e.g., "London", "New York", "Tokyo")
    
    Returns:
        Weather information as a string
    """
    # Demo implementation - replace with actual weather API
    weather_data = {
        "london": "Cloudy, 15°C (59°F), Light rain expected",
        "new york": "Sunny, 22°C (72°F), Clear skies",
        "tokyo": "Partly cloudy, 18°C (64°F), Pleasant conditions",
        "paris": "Overcast, 13°C (55°F), Chance of rain",
        "sydney": "Sunny, 25°C (77°F), Perfect beach weather",
    }
    
    location_lower = location.lower()
    
    # Check if we have data for this location
    if location_lower in weather_data:
        return f"Weather in {location}: {weather_data[location_lower]}"
    
    # Generic response for unknown locations
    return f"Weather in {location}: Mostly sunny, 20°C (68°F). (Note: This is demo data. Set WEATHER_API_KEY environment variable for real weather data.)"

