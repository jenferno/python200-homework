import json
import pytest

# Question 1

class ModelInfo:
    def __init__(self, path):
        with open(path, 'r') as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        return len(self.data['features'])



sample_data = {
    'features': [
        'temperature',
        'precipitation',
        'wind_speed'
    ]
}

with open('sample_model.json', 'w') as file:
    json.dump(sample_data, file)


model = ModelInfo('sample_model.json')

print('Feature count:', model.feature_count())

# Loading the file during initialization is more efficient 
# because it allows the data to be read once and reused 
# whenever `feature_count()` is called. This reduces 
# unnecessary file operations and improves the overall 
# efficiency of the program.

from dataclasses import dataclass

# Question 2

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        if self.will_rain:
            rain = 'rain expected'
        else:
            rain = 'no rain expected'

        return f'{self.date}: high {self.high_c:.1f}C, {rain}'


day1 = DayForecast('2023-06-15', 24.1, True)
day2 = DayForecast('2023-06-16', 27.5, False)

print(day1.summary())
print(day2.summary())

# The @dataclass decorator automatically generates the
# __init__ method based on the class attributes, eliminating
# the need to define the method manually.



# Question 3

def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError('Invalid temperature value')

def test_parse_temperature_valid():
    assert parse_temperature('24.1') == pytest.approx(24.1)

def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match='temperature'):
        parse_temperature('hot')

# Using `match` allows pytest to verify that the error message
# contains the expected text, ensuring that the message is
# accurate and clearly communicates the issue.