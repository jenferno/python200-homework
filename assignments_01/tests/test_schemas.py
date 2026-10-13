import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit.schemas import WeatherResponse


def test_real_weather_response_validates():
    '''Test that the real weather JSON validates successfully.'''
    # Using the test file's location makes this path reliable no matter
    # which directory pytest is run from. A plain relative path depends
    # on the current working directory and could point to the wrong place.
    json_path = Path(__file__).parent.parent / 'weather_raw.json'

    with open(json_path, 'r') as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude_raises():
    '''Test that a latitude outside the valid range is rejected.'''
    data = {
        'latitude': 200.0,
        'longitude': -80.0,
        'timezone': 'CST',
        'elevation': 200.0,
        'hourly': {
            'time': ['2026-04-08T00:00'],
            'temperature_2m': [15.0],
            'precipitation': [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_hourly_lengths_raise():
    '''Test that hourly lists with different lengths are rejected.'''
    data = {
        'latitude': 35.0,
        'longitude': -80.0,
        'timezone': 'CST',
        'elevation': 200.0,
        'hourly': {
            'time': [
                '2026-04-08T00:00',
                '2026-04-08T01:00',
                '2026-04-08T02:00',
            ],
            'temperature_2m': [15.0, 16.0],
            'precipitation': [0.0, 0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature_raises():
    '''Test that a null temperature value is rejected.'''
    data = {
        'latitude': 35.0,
        'longitude': -80.0,
        'timezone': 'CST',
        'elevation': 200.0,
        'hourly': {
            'time': [
                '2026-04-08T00:00',
                '2026-04-08T01:00',
            ],
            'temperature_2m': [15.0, None],
            'precipitation': [0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)