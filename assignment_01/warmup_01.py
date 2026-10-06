from dataclasses import dataclass, FrozenInstanceError, field
from pydantic import BaseModel, Field, ValidationError, model_validator
import pytest

# --- Classes --- # Q1

# Q1

class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    # Q2
    def __repr__(self):
        return (
            f"Thermometer(location='{self.location}', "
            f'n_readings={len(self.readings)}, '
            f'average={self.average()})'
        )


# Q1 : Create and test a Thermometer

my_thermometer = Thermometer('Houston')

my_thermometer.add(24)
my_thermometer.add(33)
my_thermometer.add(23)
my_thermometer.add(26)

print('Average:', my_thermometer.average())
print('Hottest:', my_thermometer.hottest())

# AVERAGE() needs to handle the empty case 
# because trying to calculate an average 
# with no values can result in an error 
# instead of a meaningful result.


# Q2 - Test __repr__

print(my_thermometer)

second_thermometer = Thermometer('Rosenberg', [28, 25, 28, 26])

print([my_thermometer, second_thermometer])

# Without __repr__, Python displays a default object address, 
# which is unhelpful for identifying the object's data while 
# debugging.

# Q3

class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [
            reading
            for reading in thermometer.readings
            if reading > self.threshold
        ]


alert_one = TemperatureAlert(26.0)
alert_two = TemperatureAlert(29.0)

print('Above 26:', alert_one.breaches(my_thermometer))
print('Above 29:', alert_two.breaches(my_thermometer))

# Storing the threshold on TemperatureAlert keeps the 
# rule in one place, so every thermometer can use the 
# same threshold without passing it into breaches() 
# each time. This makes the code more consistent, 
# easier to maintain, and simpler to manage when 
# checking twenty thermometers.


# --- Dataclasses, Type Hints, and Docstrings ---

# Q1

@dataclass(frozen=True)
class Station:
    '''Represents a weather station and its location information.'''

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station(
    'KHOU13',
    'Houston Station',
    29.65,
    -95.29,
    13.2
)

station_b = Station(
    'KIAH11',
    'George Bush Station',
    29.98,
    95.36,
    27.5
)

print('Stations equal:', station_a == station_b)

# # The result is different because the updated 
# class changes the behavior, while the original 
# hand-written class would have returned the 
# previous value.

# Q2

try:
    station_a.name = 'New Station Name'
except FrozenInstanceError as e:
    print('Frozen error:', e)


station_c = Station(
    'USC4333',
    'League City Station',
    29.47,
    -95.08,
    5.8
)

station_set = {station_a, station_b, station_c}

print('Number of unique stations:', len(station_set))

# frozen = True makes instances hashable, so they can be safely used 
# as dictionary keys or stored in sets. This is useful when you want 
# each object’s values to remain fixed and reliably identify that 
# object.
# Since station_a and station_b have identical values, the set treats
# them as duplicates, so the set contains only two unique stations.

# Q3

# Error from using stations: list[Station] = []:
# Traceback (most recent call last):
#  File "c:\Users\jenni\python200-homework\assignments_01\warmup_01.py", line 154, in <module>
#    @dataclass
#       ^^^^^^^^^
#  File "C:\Users\jenni\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1450, in dataclass
#    return wrap(cls)
#  File "C:\Users\jenni\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1440, in wrap
#    return _process_class(cls, init, repr, eq, order, unsafe_hash,
#                          frozen, match_args, kw_only, slots,
#                          weakref_slot)
#  File "C:\Users\jenni\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 1066, in _process_class
#    cls_fields.append(_get_field(cls, name, type, kw_only))
#                      ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "C:\Users\jenni\AppData\Local\Python\pythoncore-3.14-64\Lib\dataclasses.py", line 917, in _get_field
#    raise ValueError(f'mutable default {type(f.default)} for field '
#                     f'{f.name} is not allowed: use default_factory')
#ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
#(python200-homework) 
#
# Python refuses [] as the default because lists are mutable.
# Using the same list across multiple StationBatch objects could
# cause changes in one object which will affect another. 
# default_factory = list ensures that each object receives its own 
# new empty list.

@dataclass
class StationBatch:
    '''Represents a group of weather stations in a region.'''

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        '''Add a station to the batch.'''
        self.stations.append(station)

    def highest(self) -> Station | None:
        '''Return the station with the highest elevation, or None if empty.'''
        if not self.stations:
            return None

        return max(self.stations, key=lambda station: station.elevation)

batch = StationBatch('South')

batch.add(station_a)
batch.add(station_b)
batch.add(station_c)

print('Stations in batch:', batch.stations)
print('Highest station:', batch.highest())

# --- Pydantic ---

# Q1

class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode='after')
    def check_failed_sensor(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError('Possible failed sensor')
        return self
    


valid_reading = Reading(
    station_id='KHOU13 ',
    timestamp='2026-10-03T12:00:00',
    temperature_c=34,
    humidity=85.0
)

print('Valid reading:', valid_reading)

# Q2

# 1. Missing required field
try:
    Reading(
        station_id='KHOU13',
        temperature_c=32,
        humidity=85.0
    )
except ValidationError as e:
    print('Missing field error:')
    print(e)


# 2. Temperature is too high
try:
    Reading(
        station_id='KHOU13',
        timestamp='2026-10-03T12:00:00',
        temperature_c=150.0,
        humidity=75.0
    )
except ValidationError as e:
    print('Temperature error:')
    print(e)


# 3. Humidity cannot be converted to a number
try:
    Reading(
        station_id='KHOU13',
        timestamp='2026-10-03T12:00:00',
        temperature_c=30.0,
        humidity='very humid'
    )
except ValidationError as e:
    print('Humidity error:')
    print(e)


# Test Pydantic type conversion
converted_reading = Reading(
    station_id='KHOU13',
    timestamp='2026-10-03T12:00:00',
    temperature_c='24.5',
    humidity=40
)

print('Converted reading:', converted_reading)
print('Temperature type:', type(converted_reading.temperature_c))
print('Humidity type:', type(converted_reading.humidity))

# Pydantic accepts '24.5' because the string can be converted into a float.
# It rejects "very humid" because that string cannot be converted into a float.
# Pydantic automatically converts compatible values to the expected type
# and raises a validation error when the conversion is not possible.

# Q3

try:
    Reading(
        station_id='1',
        temperature_c='twenty degrees celius',
        humidity=50.0
    )
except ValidationError as e:
    for error in e.errors():
        print('Location:', error['loc'])
        print('Message:', error['msg'])


# Three errors were reported: the station_id is too short,
# the timestamp is missing, and temperature_c is not a valid number.
# Reporting all the errors at once is more useful because we can see
# everything that needs to be fixed instead of fixing one error at a time.

# Q4

valid_sensor_reading = Reading(
    station_id='KHOU13',
    timestamp='2026-10-03T12:00:00',
    temperature_c=25.0,
    humidity=65.0
)

print('Valid sensor reading:', valid_sensor_reading)


try:
    bad_sensor_reading = Reading(
        station_id='KHOU13',
        timestamp='2026-10-03T12:00:00',
        temperature_c=-45.0,
        humidity=0.0
    )
except ValidationError as e:
    print('Failed sensor error:')
    print(e)


# This rule cannot be handled with Field constraints alone 
# because it compares the values of two fields: humidity and 
# temperature_c. Field constraints only validate individual 
# fields separately.


# --- pytest ---

# Q1

def celsius_to_fahrenheit(celsius: float) -> float:
    '''Convert a temperature from Celsius to Fahrenheit.'''
    return (celsius * 9 / 5) + 32


def test_celsius_to_fahrenheit():
    '''Test Celsius to Fahrenheit conversions.'''
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# pytest.approx is used because floating-point calculations can have
# tiny rounding differences, so an exact == comparison may fail.


# Q2

def mean(values: list[float]) -> float:
    '''Return the mean of a list of values.'''
    if not values:
        raise ValueError('values cannot be empty')

    return sum(values) / len(values)


def test_mean_of_empty_raises():
    '''Test that mean raises ValueError for an empty list.'''
    with pytest.raises(ValueError, match="empty"):
        mean([])

# pytest.raises(ValueError) checks that a ValueError was raised.
# Using match= also verifies that the error message contains the expected
# word, helping confirm that the error occurred for the correct reason.

# Q3

@pytest.mark.parametrize(
    'values, expected',
    [
        ([4.0], 4.0),
        ([8.0, 12.0, 16.0], 20.0),
        ([-4.0, -8.0, -12.0], -16.0),
        ([-8.0, 8.0], 0.0),
    ],
)
def test_mean_values(values, expected):
    '''Test mean with several different lists of values.'''
    assert mean(values) == pytest.approx(expected)

# 6 passed in 0.18s

# One parametrized test is better than four nearly identical test functions
# because it avoids repeated code making the tests easier to read and to maintain.

# Q4

# Failure output after changing 9 / 5 to 9 / 4:
# assert celsius_to_fahrenheit(35) == 95
# E       assert 203.0 == 95
# E        +  where 203.0 = celsius_to_fahrenheit(35)
# FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 203.0 == 95
# 1 failed, 5 passed in 0.23s

# Pytest showed that when the input was 35, the function returned 203.0,
# but the expected value was 95. This is more helpful than simply reporting 
# an assertion failure because it shows both the actual and expected values, 
# making the mistake easier to identify.