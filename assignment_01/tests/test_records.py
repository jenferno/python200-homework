from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import WeatherResponse


def make_response() -> WeatherResponse:
    '''Create a small validated weather response for testing.'''
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
            'temperature_2m': [10.0, 15.0, 20.0],
            'precipitation': [0.0, 1.5, 0.5],
        },
    }

    return WeatherResponse.model_validate(data)


def test_to_readings_preserves_count_and_order():
    '''Test that one reading is created per hour in the correct order.'''
    response = make_response()

    readings = to_readings(response)

    assert len(readings) == 3
    assert readings[0].timestamp == '2026-04-08T00:00'
    assert readings[-1].timestamp == '2026-04-08T02:00'


def test_reading_values_match_input_indexes():
    '''Test that values at the same input index stay together.'''
    response = make_response()

    readings = to_readings(response)

    assert readings[1].timestamp == response.hourly.time[1]
    assert readings[1].temperature_c == response.hourly.temperature_2m[1]
    assert readings[1].precipitation_mm == response.hourly.precipitation[1]


def test_identical_hourly_readings_are_equal():
    '''Test that identical HourlyReading objects compare equal.'''
    reading_one = HourlyReading(
        timestamp='2026-04-08T00:00',
        temperature_c=15.0,
        precipitation_mm=1.0,
    )

    reading_two = HourlyReading(
        timestamp='2026-04-08T00:00',
        temperature_c=15.0,
        precipitation_mm=1.0,
    )

    assert reading_one == reading_two