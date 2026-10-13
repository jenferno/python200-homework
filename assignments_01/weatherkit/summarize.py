from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    '''Represents a summary of weather observations for one day.'''

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        '''Return the difference between the maximum and minimum temperature.'''
        return self.temp_max - self.temp_min
# Deliberate-break check: I changed temp_range() from
# subtraction to addition. The parametrized test_temp_range tests caught
# the change, resulting in 4 failed tests and 12 passed tests.

class DailyAggregator:
    '''Groups hourly weather readings into daily summaries.'''

    def __init__(self, min_hours: int = 24):
        '''Initialize the aggregator with the minimum required hours.

        Args:
            min_hours: Minimum number of hourly observations required
                for a day to be included in the summaries.
        '''
        self.min_hours = min_hours

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        '''Group hourly readings into daily weather summaries.

        Args:
            readings: A list of hourly weather readings.

        Returns:
            A list of daily summaries sorted by date. Days with fewer
            than the minimum required observations are excluded.
        '''
        grouped: dict[str, list[HourlyReading]] = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in grouped:
                grouped[date] = []

            grouped[date].append(reading)

        summaries = []

        for date, daily_readings in grouped.items():
            if len(daily_readings) < self.min_hours:
                continue

            temperatures = [
                reading.temperature_c
                for reading in daily_readings
            ]

            precipitation = [
                reading.precipitation_mm
                for reading in daily_readings
            ]

            summary = DailySummary(
                date=date,
                temp_max=max(temperatures),
                temp_min=min(temperatures),
                precipitation_sum=sum(precipitation),
                hours_observed=len(daily_readings),
            )

            summaries.append(summary)

        return sorted(summaries, key=lambda summary: summary.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        '''Return dates that have fewer than the required observations.

        Args:
            readings: A list of hourly weather readings.

        Returns:
            A sorted list of dates that do not meet the minimum number
            of hourly observations.
        '''
        counts: dict[str, int] = {}

        for reading in readings:
            date = reading.timestamp[:10]
            counts[date] = counts.get(date, 0) + 1

        incomplete = [
            date
            for date, count in counts.items()
            if count < self.min_hours
        ]

        return sorted(incomplete)