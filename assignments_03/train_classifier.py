# Part 2: Weather Classifier Mini-Project

# Task 1: Fetch the Data
# City chosen: Houston, Texas, USA

import requests
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
import json
import joblib
import sklearn
from pathlib import Path



url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    'latitude': 29.76,
    'longitude': -95.37,
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
    'timezone': 'America/Chicago',
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

df = pd.DataFrame(response.json()['daily'])

df['date'] = pd.to_datetime(df['time'])
df = df.drop('time', axis=1)

print('Task 1: Houston Weather Data')
print('Dataset shape:', df.shape)
print(df.head())



# Task 2: Engineer the Label

df['good_for_running'] = (
    (df['temperature_2m_max'] >= 7) &
    (df['temperature_2m_max'] <= 26) &
    (df['temperature_2m_min'] >= 0) &
    (df['precipitation_sum'] < 3) &
    (df['wind_speed_10m_max'] < 30)
).astype(int)

print('\nTask 2: Class Balance')

print(df['good_for_running'].value_counts())

good_days = df['good_for_running'].sum()
total_days = len(df)

good_fraction = good_days / total_days

print(f'Good running days: {good_days}')
print(f'Skip running days: {total_days - good_days}')
print(f'Fraction of good days: {good_fraction:.2%}')

# About 28.49% of the days in the dataset were 
# classified as suitable for running, while 71.51% were 
# classified as days to skip. This distribution seems 
# reasonable for Houston, where hot summers, rainfall, 
# and occasional cold weather can affect outdoor running 
# conditions. The dataset is also imbalanced, with 104 
# good running days compared to 261 days to skip. 
# Therefore, I will evaluate the model using precision, 
# recall, and F1-score in addition to accuracy to gain a 
# more comprehensive understanding of its performance.


# Task 3: Build and Evaluate a KNN Classifier

features = [
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    'wind_speed_10m_max'
]

X = df[features]
y = df['good_for_running']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

k_values = [1, 3, 5, 7, 9, 11, 15, 21]

cv_results = {}

print('\nTask 3: KNN Cross-Validation Results')

for k in k_values:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy'
    )

    mean_score = scores.mean()
    cv_results[k] = mean_score

    print(f'k={k}: {mean_score:.4f}')

best_k = max(cv_results, key=cv_results.get)

print(f'\nBest k: {best_k}')

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])

knn_pipeline.fit(X_train, y_train)

knn_predictions = knn_pipeline.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print('\nKNN Test Accuracy:', knn_accuracy)

print('\nKNN Classification Report:')
print(classification_report(y_test, knn_predictions))


# Task 4: Build and Evaluate Logistic Regression

logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

logreg_pipeline.fit(X_train, y_train)

logreg_predictions = logreg_pipeline.predict(X_test)

logreg_accuracy = accuracy_score(y_test, logreg_predictions)
logreg_precision = precision_score(y_test, logreg_predictions)
logreg_recall = recall_score(y_test, logreg_predictions)
logreg_f1 = f1_score(y_test, logreg_predictions)

print('\nTask 4: Logistic Regression Results')

print(f'Accuracy: {logreg_accuracy:.4f}')
print(f'Precision: {logreg_precision:.4f}')
print(f'Recall: {logreg_recall:.4f}')
print(f'F1-score: {logreg_f1:.4f}')

print('\nLogistic Regression Classification Report:')
print(classification_report(y_test, logreg_predictions))

print('\nLogistic Regression Feature Coefficients:')

coefficients = logreg_pipeline.named_steps['classifier'].coef_[0]

for feature, coefficient in zip(features, coefficients):
    print(f"{feature}: {coefficient:.4f}")


# Approximately 28.49% of the days in the dataset were classified as 
# suitable for running, while 71.51% were classified as days to skip. 
# This distribution seems reasonable for Atlanta, where hot summers, 
# rainfall, and occasional cold weather can affect outdoor running 
# conditions. Since the dataset is imbalanced, with 104 good running 
# days compared to 261 days to skip, I will evaluate the model using 
# precision, recall, and F1-score in addition to accuracy.
#
# The Logistic Regression coefficients show that precipitation has 
# the strongest negative influence on the model's predictions, with 
# a coefficient of -3.2852. Maximum temperature (-1.4959), minimum 
# temperature (-1.0788), and maximum wind speed (-0.3626) also have 
# negative coefficients. These results suggest that higher values 
# for these features are associated with a lower likelihood of a 
# day being classified as suitable for running, assuming the other 
# features remain constant. However, the coefficients should be 
# interpreted in the context of the feature scaling and encoding 
# used in the model.



# Task 5: Compare and Choose

print('\nTask 5: Model Comparison')
print(f'KNN Accuracy: {knn_accuracy:.4f}')
print(f'Logistic Regression Accuracy: {logreg_accuracy:.4f}')

# KNN achieved a test accuracy of 98.63%, outperforming Logistic 
# Regression, which achieved 97.26%. For a running application, 
# I would prioritize precision because a false positive could 
# result in recommending a run when weather conditions are 
# unsuitable or potentially unsafe. In contrast, a false 
# negative would mean missing an opportunity to recommend 
# running on a favorable day.
# 
# Based on the test accuracy results, KNN performed slightly 
# better than Logistic Regression on this dataset. However, 
# I would also compare their precision, recall, and F1-scores 
# before making a final decision about deployment. Logistic 
# Regression has the advantage of being easier to interpret 
# through its feature coefficients and provides more informative 
# probability estimates in many cases. KNN with k=1 can produce 
# extreme probability estimates of 0 or 1 based on its single 
# nearest neighbor, making its confidence estimates less informative. 
# Despite this limitation, KNN is a promising candidate because 
# it achieved the highest test accuracy.

chosen_pipeline = knn_pipeline
chosen_accuracy = knn_accuracy
chosen_f1 = f1_score(y_test, knn_predictions)



# Task 6: Save the Model

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

model_path = models_dir / 'weather_classifier.pkl'
metadata_path = models_dir / 'weather_classifier_metadata.json'

# Save the complete fitted pipeline
joblib.dump(chosen_pipeline, model_path)

# Create the metadata dictionary
metadata = {
    'city': 'Houston, Texas',
    'model_type': f'KNeighborsClassifier (k={best_k})',
    'python_version': sys.version.split()[0],
    'scikit_learn_version': sklearn.__version__,
    'features': features,
    'label_rule': (
        'good_for_running = 1 when maximum temperature is '
        'between 7 and 26 degrees Celsius, minimum temperature '
        'is at least 0 degrees Celsius, precipitation is under '
        '3 mm, and maximum wind speed is under 30 km/h. '
        'Otherwise, good_for_running = 0.'
    ),
    'test_accuracy': float(chosen_accuracy),
    'test_f1': float(chosen_f1)
}

# Save metadata as a JSON file
with open(metadata_path, 'w', encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print('\nTask 6: Model Saved Successfully')
print(f'Model saved to: {model_path}')
print(f'Metadata saved to: {metadata_path}')