from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

# The four weather features used to train the model
FEATURES = [
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    'wind_speed_10m_max',
]


@dataclass
class Prediction:
    '''Stores a running weather prediction and its probability.'''

    label: str
    probability: float

# Task 2 : The WeatherClassifier Class
class WeatherClassifier:
    '''Loads a saved weather model and predicts running conditions.'''

    def __init__(self, model_path: str | Path):
        '''Load the trained model.

        Args:
            model_path: Location of the saved model.

        Raises:
            FileNotFoundError: If the model file does not exist.
        '''
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f'No model file at {model_path}. '
                'Train and save a model first.'
            )

        self.model = joblib.load(model_path)
        self.features = FEATURES

    def predict(self, days: list[dict]) -> list[Prediction]:
        '''Predict whether each day is good for running.

        Args:
            days: A list of dictionaries containing weather features.

        Returns:
            A list of Prediction objects in the same order as the days.

        Raises:
            ValueError: If required weather features are missing.
        '''
        if not days:
            return []

        frame = pd.DataFrame(days)

        missing = [
            feature
            for feature in self.features
            if feature not in frame.columns
        ]

        if missing:
            raise ValueError(
                f'missing required features: {missing}'
            )

        X = frame[self.features]

        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

# To verify the test's effectiveness, I temporarily modified the
# function to return "skip" for every day. This caused
# `test_weather_labels[day0-good]` to fail as expected,
# confirming that the test correctly identifies incorrect label
# conversions. I then restored the original label conversion logic.


        return [
            Prediction(
                'good' if int(label) == 1 else 'skip',
                float(probability)
            )
            for label, probability in zip(labels, probabilities)
        ]