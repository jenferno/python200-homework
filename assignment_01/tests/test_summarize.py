import pytest

from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator


@pytest.fixture
def sample_readings():
    '''Provide hourly readings spanning two calendar dates.'''
    return [
        HourlyReading('2026-04-08T00:00', 10.0, 0.5),
        HourlyReading('2026-04-08T01:00', 15.0, 1.0),
        HourlyReading('2026-04-08T02:00', 20.0, 0.5),
        HourlyReading('2026-04-09T00:00', 5.0, 0.2),
        HourlyReading('2026-04-09T01:00', 12.0, 0.3),
    ]


def test_grouping_produces_two_summaries(sample_readings):
    '''Test that readings are grouped by calendar date.'''
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == '2026-04-08'
    assert summaries[1].date == '2026-04-09'


def test_max_and_min_temperatures(sample_readings):
    '''Test daily maximum and minimum temperatures.'''
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].temp_max == 20.0
    assert summaries[0].temp_min == 10.0


def test_precipitation_sum(sample_readings):
    '''Test that daily precipitation is added correctly.'''
    aggregator = DailyAggregator(min_hours=1)

    summaries = aggregator.summarize(sample_readings)

    assert summaries[0].precipitation_sum == pytest.approx(2.0)


def test_incomplete_day_is_dropped(sample_readings):
    '''Test that days below min_hours are excluded and reported.'''
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)
    incomplete = aggregator.incomplete_days(sample_readings)

    assert len(summaries) == 1
    assert summaries[0].date == '2026-04-08'
    assert '2026-04-09' in incomplete


def test_lower_min_hours_keeps_day(sample_readings):
    '''Test that lowering min_hours allows an incomplete day to be kept.'''
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    dates = [summary.date for summary in summaries]

    assert '2026-04-09' in dates


@pytest.mark.parametrize(
    'temp_max, temp_min, expected_range',
    [
        (20.0, 10.0, 10.0),
        (15.0, 5.0, 10.0),
        (5.0, -5.0, 10.0),
        (10.0, 10.0, 0.0),
    ],
)
def test_temp_range(temp_max, temp_min, expected_range):
    '''Test temperature range calculations with several values.'''
    from weatherkit.summarize import DailySummary

    summary = DailySummary(
        date='2026-04-08',
        temp_max=temp_max,
        temp_min=temp_min,
        precipitation_sum=0.0,
        hours_observed=24,
    )

    assert summary.temp_range() == pytest.approx(expected_range)