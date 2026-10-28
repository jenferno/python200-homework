# Task 4 : The Test Suite

import pytest
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from weather_model import WeatherClassifier, Prediction, FEATURES


@pytest.fixture
def model_path(tmp_path):
    '''Train and save a small model for testing.'''

    good = [{
        'temperature_2m_max': 18,
        'temperature_2m_min': 9,
        'precipitation_sum': 0.0,
        'wind_speed_10m_max': 10,
    }] * 15

    bad = [{
        'temperature_2m_max': 2,
        'temperature_2m_min': -6,
        'precipitation_sum': 12.0,
        'wind_speed_10m_max': 40,
    }] * 15

    X = pd.DataFrame(good + bad)[FEATURES]
    y = [1] * 15 + [0] * 15

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000)),
    ])

    model.fit(X, y)

    path = tmp_path / 'test_model.pkl'
    joblib.dump(model, path)

    return path


@pytest.fixture
def clf(model_path):
    '''Return a classifier using the test model.'''
    return WeatherClassifier(model_path)


@pytest.mark.parametrize(
    'day, expected',
    [
        ({
            'temperature_2m_max': 18,
            'temperature_2m_min': 9,
            'precipitation_sum': 0.0,
            'wind_speed_10m_max': 10,
        }, 'good'),
        ({
            'temperature_2m_max': 2,
            'temperature_2m_min': -6,
            'precipitation_sum': 12.0,
            'wind_speed_10m_max': 40,
        }, 'skip'),
    ],
)
def test_weather_labels(clf, day, expected):
    result = clf.predict([day])
    assert result[0].label == expected


def test_probability_range(clf):
    day = {
        'temperature_2m_max': 15,
        'temperature_2m_min': 8,
        'precipitation_sum': 1.0,
        'wind_speed_10m_max': 12,
    }

    result = clf.predict([day])

    assert 0 <= result[0].probability <= 1


def test_three_predictions_in_order(clf):
    good = {
        'temperature_2m_max': 18,
        'temperature_2m_min': 9,
        'precipitation_sum': 0.0,
        'wind_speed_10m_max': 10,
    }

    bad = {
        'temperature_2m_max': 2,
        'temperature_2m_min': -6,
        'precipitation_sum': 12.0,
        'wind_speed_10m_max': 40,
    }

    results = clf.predict([good, bad, good])

    assert len(results) == 3
    assert all(isinstance(r, Prediction) for r in results)
    assert [r.label for r in results] == [
        'good', 'skip', 'good'
    ]


def test_empty_list(clf):
    assert clf.predict([]) == []


def test_missing_features(clf):
    with pytest.raises(
        ValueError,
        match='missing required features'
    ):
        clf.predict([
            {'temperature_2m_max': 18}
        ])


def test_missing_model(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match='No model file'
    ):
        WeatherClassifier(tmp_path / 'missing.pkl')


def test_prediction_type(clf):
    day = {
        'temperature_2m_max': 18,
        'temperature_2m_min': 9,
        'precipitation_sum': 0.0,
        'wind_speed_10m_max': 10,
    }

    result = clf.predict([day])

    assert isinstance(result[0], Prediction)