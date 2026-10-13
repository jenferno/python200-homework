from dataclasses import dataclass

from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    '''Represents one hourly weather reading.

    Temperature is measured in degrees Celsius.
    Precipitation is measured in millimeters.
    '''

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    '''Convert a validated weather response into hourly readings.

    Args:
        response: A validated WeatherResponse containing hourly weather data.

    Returns:
        A list of HourlyReading objects in the same order as the API response.
    '''
    readings = []

    for timestamp, temperature, precipitation in zip(
        response.hourly.time,
        response.hourly.temperature_2m,
        response.hourly.precipitation,
    ):
        reading = HourlyReading(
            timestamp=timestamp,
            temperature_c=temperature,
            precipitation_mm=precipitation,
        )
        readings.append(reading)

    return readings


# HourlyReading is a dataclass because it represents internal, already-validated 
# weather data, while WeatherResponse is a Pydantic model because it sits at the 
# boundary where external API data enters the program and needs validation and
# type conversion.