# Week 2 Mini-Project: Predicting the Daily High
# City: Houston, Texas
# Year: 2023

import requests
import pandas as pd

# --- Task 1: Fetch the Data ---

# Open-Meteo historical weather API
url = 'https://archive-api.open-meteo.com/v1/archive'


params = {
'latitude': 29.76,
'longitude': -95.37,
'start_date': '2023-01-01',
'end_date': '2023-12-31',
'daily': [
'temperature_2m_max',
'temperature_2m_min',
'precipitation_sum',
'wind_speed_10m_max',
],
'timezone': 'America/Chicago',
}


response = requests.get(url, params=params, timeout=30)

response.raise_for_status()

df = pd.DataFrame(response.json()['daily'])

df['date'] = pd.to_datetime(df['time'])

df = df.drop('time', axis=1)

print('\n--- Task 1: Weather Dataset ---')
print('Location: Houston, Texas')
print('Dataset shape:', df.shape)

print('\nFirst five rows:')
print(df.head())

# The city I picked in the closest major city to where I live.

# --- Task 2: Look at the Features ---

import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).resolve().parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)

print('\n--- Task 2: Explore the Data ---')
print('\nSummary statistics:')
print(df.describe())

print('\nMissing values:')
print(df.isnull().sum())

weather_cols = [
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    'wind_speed_10m_max',
]

df = df.dropna(subset=weather_cols).copy()

correlations = (
    df.select_dtypes(include='number')
    .corr()['temperature_2m_max']
    .sort_values(ascending=False)
)

print('\nCorrelations with daily high:')
print(correlations)

plt.scatter(
    df['temperature_2m_min'],
    df['temperature_2m_max'],
    alpha=0.6
)

plt.title('Houston Daily High vs Low (2023)')
plt.xlabel('Daily Low Temperature (°C)')
plt.ylabel('Daily High Temperature (°C)')

plt.savefig(output_dir / 'houston_high_vs_low.png')
plt.close()

print('\nWeather scatter plot saved successfully!')

# The scatter plot demonstrates a positive linear trend where 
# daily low temperatures are typically associated with daily high 
# temperatures.


# --- Task 3: Engineer a Feature ---

df['is_summer'] = df['date'].dt.month.isin([6, 7, 8]).astype(int)

summer_counts = df['is_summer'].value_counts()

print('\n--- Task 3: Summer Feature ---')
print('Summer days:', summer_counts.get(1, 0))
print('Non-summer days:', summer_counts.get(0, 0))

# Yes, I expect "is_summer" to help predict the daily high 
# temperatures because summers in Houston are typically very hot.


# --- Task 4: Baseline Model ---

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X_baseline = df[['temperature_2m_min']].values

y = df['temperature_2m_max'].values

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline,
    y,
    test_size=0.2,
    random_state=42
)

baseline_model = LinearRegression()

baseline_model.fit(X_train_base, y_train_base)

y_pred_base = baseline_model.predict(X_test_base)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test_base, y_pred_base)
)

baseline_r2 = r2_score(y_test_base, y_pred_base)

print('\n--- Task 4: Baseline Model ---')
print('Slope:', baseline_model.coef_[0])
print('RMSE:', baseline_rmse)
print('R-squared (R²):', baseline_r2)

# RMSE is measured in degrees Celsius.
# The typical predication is on average 
# about 2.9°C off from the actual temperature.


# --- Task 5: Full Model ---

feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X_full = df[feature_cols].values
y_full = df["temperature_2m_max"].values

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=42
)

full_model = LinearRegression()

full_model.fit(X_train_full, y_train_full)

y_pred_full = full_model.predict(X_test_full)

train_r2_full = full_model.score(X_train_full, y_train_full)
test_r2_full = full_model.score(X_test_full, y_test_full)

rmse_full = np.sqrt(
    mean_squared_error(y_test_full, y_pred_full)
)

print('\n--- Task 5: Full Model ---')
print('Training R²:', train_r2_full)
print('Testing R²:', test_r2_full)
print('RMSE:', rmse_full)

print('\nBaseline R²:', baseline_r2)
print('Full Model R²:', test_r2_full)
print('Improvement in R²:', test_r2_full - baseline_r2)

# The R² increased from 0.881 to 0.898 which means that 
# the model improved in predicting the daily high 
# temperature. The closer the number is to 1 indicates 
# a better fit. By adding the 4 features, it provides 
# the model with more information that allows it to 
# accurately predict the daily high temperature.


print('\nFeature Coefficients:')

for name, coef in zip(feature_cols, full_model.coef_):
    print(f'{name:22s}: {coef:+.3f}')

# A positive coefficient can be interpreted as making 
# the most impactful in predicting the daily high 
# temperature more accurately, while a negative 
# coefficient would be interpreted as the least 
# impactful in predicting the daily high temperature 
# less accurately. 

# Task 5 Interpretation : 
# Comparing the train R² and the test R², they are close
# in value with only a difference of 0.0175. 
# 
# This lets us know that the first training model was 
# pretty accurate and generalizes the high temperatures 
# in Houston. This could also mean that the temperatures in 
# Houston are fairly consistent in the relationship of 
# the features between high temperatures.



# --- Task 6: Evaluate and Summarize ---

plt.scatter(
    y_pred_full,
    y_test_full,
    alpha=0.7,
    label='Test Days'
)

min_temp = min(y_pred_full.min(), y_test_full.min())
max_temp = max(y_pred_full.max(), y_test_full.max())

plt.plot(
    [min_temp, max_temp],
    [min_temp, max_temp],
    color='red',
    linestyle='--',
    label='Perfect Prediction'
)

plt.title('Predicted vs Actual (Full Model)')
plt.xlabel('Predicted High Temperature (°C)')
plt.ylabel('Actual High Temperature (°C)')
plt.legend()

plt.savefig(output_dir / 'predicted_vs_actual_high.png')
plt.close()

print('\n--- Task 6: Evaluation ---')
print('Predicted vs Actual graph saved successfully!')


# --- Task 6: Final Project Summary ---
#
# The dataset contains 365 days of Houston weather
# data from 2023. I used 292 days to train the model
# and the remaining 73 days to test its predictions.
#
# The full model has an RMSE of 2.68°C which indicates 
# that the prediction errors are typically around 2.68°C. 
# Its R² of 0.898 also explains about 89.8% of the variation 
# in daily high temperatures in the test data.
#
# The is_summer feature has the largest absolute
# coefficient (+1.844). Its positive direction means
# the model predicts summer days to be about 1.84°C
# warmer when other features remain constant.
#
# One surprising result is that the minimum daily 
# temperature has a negative coefficient of -0.098. 
# This describes the relationship between the minimum 
# and daily high temperature as for every 1°C increase 
# in daily minimum temperature the predicted daily high 
# will decrease by -0.098°C. Even though the coefficient 
# is relatively small, I assumed that the relationship 
# overall would be positive.