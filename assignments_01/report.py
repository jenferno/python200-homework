import json
from pathlib import Path

from weatherkit import WeatherResponse
from weatherkit.records import to_readings
from weatherkit.summarize import DailyAggregator


def main() -> None:
    '''Load, validate, summarize, and report the weather data.'''
    json_path = Path(__file__).parent / 'weather_raw.json'

    with open(json_path, 'r') as file:
        raw_data = json.load(file)

    weather = WeatherResponse.model_validate(raw_data)

    readings = to_readings(weather)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    print(
        f"{'Date':<12}"
        f"{'High':>8}"
        f"{'Low':>8}"
        f"{'Precip':>10}"
        f"{'Range':>10}"
    )
    print('-' * 48)

    for summary in summaries:
        print(
            f'{summary.date:<12}'
            f'{summary.temp_max:>8.1f}'
            f'{summary.temp_min:>8.1f}'
            f'{summary.precipitation_sum:>10.1f}'
            f'{summary.temp_range():>10.1f}'
        )

    if incomplete:
        print('\nWarning: Incomplete days dropped:', ', '.join(incomplete))
    else:
        print('\nWarning: No incomplete days were dropped.')


# Without this guard, the main() function would run automatically whenever another 
# Python file imported report.py, causing the report to be generated and printed 
# even when the importing file only needed to use its functions. 
# The if __name__ == "__main__": guard ensures that main() runs only when report.py 
# is executed directly, while still allowing other programs to import and reuse its 
# functions without automatically running the report.

if __name__ == '__main__':
    main()


# Reflection Questions
#
# 1. WeatherResponse currently rejects the entire file if even one temperature 
# is null. This can be helpful if the pipeline needs complete data before it 
# can continue, because it prevents incomplete information from being used. 
# However, if a weather sensor happens to miss one reading, it might be better 
# to keep the other valid readings instead of throwing away the entire file. 
# One way to allow this would be to change the schema from `list[float]` to 
# `list[float | None]`. The pipeline would then have to decide what to do with 
# those missing values, such as skipping them or handling them separately.

#
# 2. DailyAggregator has a default `min_hours` value of 24, meaning it expects 
# a full day of hourly readings. For example, if the pipeline runs around noon, 
# the current day might only have about 12 readings because the day is not over 
# yet. The aggregator could leave that day out because it does not have enough 
# readings, even though the data itself is not necessarily wrong. The 
# `incomplete_days()` method is useful because it shows which dates were 
# left out due to not having enough observations. This helps us understand 
# whether data is actually missing or if the day simply has not finished 
# collecting data yet.

#
# 3. Keeping `weatherkit` as a package makes the code easier to reuse in other 
# projects or pipelines. Instead of putting all the code into one large file and 
# copying it whenever we need it somewhere else, the Week 10 pipeline can import 
# the specific classes and functions it needs from `weatherkit`. For example, it 
# could import `WeatherResponse` or `DailyAggregator` and use them directly. This
#  makes the code more organized and saves time because we can reuse code we have 
# already written instead of creating it again.
