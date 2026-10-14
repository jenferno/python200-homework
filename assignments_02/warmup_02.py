# Week 2 Warmup Exercises

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression


# --- scikit-learn API ---
# Q1

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

model = LinearRegression()

model.fit(temp_min, temp_max)

predictions = model.predict([[6], [18]])

print('Slope:', model.coef_[0])
print('Intercept:', model.intercept_)
print('Predicted high when low is 6°C:', predictions[0])
print('Predicted high when low is 18°C:', predictions[1])


# Q2

x = np.array([10, 20, 30, 40, 50])

print('\nOriginal shape:', x.shape)

x_2darray = x.reshape(-1, 1)

print('Reshaped shape:', x_2darray.shape)

# Scikit-learn needs X to be 2D to explicityly distinguish between 
# the samples and features, where one represents the observation 
# and the other represents the feature.


# --- Linear Regression ---
# Provided synthetic weather dataset

rng = np.random.default_rng(42)
num_days = 120

temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)

temp_max = (
    1.0 * temp_min
    + 6
    - 4 * is_rainy
    + rng.normal(0, 2, num_days)
)


# Q1

output_dir = Path(__file__).resolve().parent / 'outputs'
output_dir.mkdir(parents=True, exist_ok=True)

plt.scatter(
    temp_min,
    temp_max,
    c=is_rainy,
    cmap='coolwarm'
)

plt.colorbar(label='Rainy Day (0 = No, 1 = Yes)')

plt.title('Daily High vs Low')
plt.xlabel('Daily Low Temperature (°C)')
plt.ylabel('Daily High Temperature (°C)')

plt.savefig(output_dir / 'high_vs_low.png')
plt.close()

print('\nScatter plot saved successfully!')

# The scatter plot shows two temperature groups based on 
# rainy and non-rainy conditions. Rainy days generally 
# have lower high temperatures than non-rainy days, but both days
# experience similar lows.


# Q2

from sklearn.model_selection import train_test_split

X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_training, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print('\nX_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_training shape:', y_training.shape)
print('y_test shape:', y_test.shape)


# Q3

from sklearn.metrics import mean_squared_error

model_one = LinearRegression()

model_one.fit(X_train, y_training)

y_pred = model_one.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = model_one.score(X_test, y_test)

print('\n--- Linear Regression Q3 ---')
print('Slope:', model_one.coef_[0])
print('Intercept:', model_one.intercept_)
print('RMSE:', rmse)
print('R-squared (R²):', r2)

# The slope tells us how much the predicted high
# temperature changes as the low temperature
# increases by 1°C.


# Q4

X_full = np.column_stack([temp_min, is_rainy])

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full,
    temp_max,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)

y_full_pred = model_full.predict(X_full_test)

r2_full = model_full.score(X_full_test, y_full_test)

print('\n--- Linear Regression Q4 ---')
print('One-feature R²:', r2)
print('Two-feature R²:', r2_full)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# The is_rainy coefficient tells us how much the
# predicted high temperature is on rainy days,
# while keeping the low temperature relatively the same.
# A negative coefficient can be interpretted that the rainy
# days tend to have lower high temperatures.


# Q5

plt.scatter(y_full_pred, y_full_test, alpha=0.7)

min_value = min(y_full_pred.min(), y_full_test.min())
max_value = max(y_full_pred.max(), y_full_test.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color='red',
    linestyle='--',
    label='Perfect Prediction'
)

plt.title('Predicted vs Actual')
plt.xlabel('Predicted High Temperature (°C)')
plt.ylabel('Actual High Temperature (°C)')
plt.legend()

plt.savefig(output_dir / 'predicted_vs_actual_warmup.png')
plt.close()

print('\nPredicted vs Actual plot saved successfully!')

# Points above the diagonal line mean the model
# underestimated the actual high temperature, 
# while points below the diagonal line mean the model
# overestimated the actual high temperature.