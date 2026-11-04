# Task 5 : The Script

'''Use the saved model to predict running weather.'''

from weather_model import WeatherClassifier


SAMPLE_DAYS = [
    {
        'temperature_2m_max': 18,
        'temperature_2m_min': 10,
        'precipitation_sum': 0.0,
        'wind_speed_10m_max': 12,
    },
    {
        'temperature_2m_max': 2,
        'temperature_2m_min': -5,
        'precipitation_sum': 12.0,
        'wind_speed_10m_max': 40,
    },
    {
        'temperature_2m_max': 22,
        'temperature_2m_min': 13,
        'precipitation_sum': 4.0,
        'wind_speed_10m_max': 18,
    },
    {
        'temperature_2m_max': 25,
        'temperature_2m_min': 15,
        'precipitation_sum': 0.0,
        'wind_speed_10m_max': 33,
    },
]


def main() -> None:
    '''Load the model and print predictions for sample days.'''

    classifier = WeatherClassifier(
        'models/weather_classifier.pkl'
    )

    results = classifier.predict(SAMPLE_DAYS)

    for day, result in zip(SAMPLE_DAYS, results):
        print(
            f'High: {day['temperature_2m_max']}C, '
            f'Low: {day['temperature_2m_min']}C, '
            f'Rain: {day['precipitation_sum']}mm, '
            f'Wind: {day['wind_speed_10m_max']} km/h '
            f'-> {result.label} '
            f'(Probability of good: {result.probability:.2f})'
        )


# This guard prevents `main()` from executing automatically when
# the script is imported into another Python file. It ensures
# that `main()` runs only when the script is executed directly.
if __name__ == '__main__':
    main()


# Task 6 : Reflection

# 1. Selecting features in a consistent, predefined order ensures
# that the model receives the correct weather values for each
# feature. If the order were changed, such as swapping wind speed
# and temperature, the model could produce inaccurate predictions.
#
# 2. Training a small model within the test fixture keeps the tests
# independent of the saved production model. This improves test
# reliability, reproducibility, and consistency across test runs.
#
# 3. My component is ready for Week 10 because it can process
# multiple weather records in a single call. Before deploying it
# in a production environment, I would implement additional input
# validation to handle missing data and invalid values more safely.