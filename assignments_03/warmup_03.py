
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib
from pathlib import Path

df = pd.read_csv("assignments_03/weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
# Q1

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Q1: Mean of each scaled training data column:')
print(X_train_scaled.mean(axis=0))

# You fit the scaler to X_train only to prevent data leakage
# from the test data.


# Q2

encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[['season']])

print('\nQ2: Encoded season shape:')
print(season_encoded.shape)

print('Season category names:')
print(encoder.get_feature_names_out())

# The season is 4 columns because the
# dataset contains four seasons: 
# fall, spring, summer, winter.


# Q3

features = numeric + ['season']

X_all = df[features]

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['season'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train_all, y_train_all)

y_pred = pipeline.predict(X_test_all)

accuracy = accuracy_score(y_test_all, y_pred)

print('\nQ3: Pipeline test accuracy:')
print(accuracy)

# The pipeline automatically standardizes numerical features, 
# converts the season variable into encoded values, and uses 
# the processed data to train the Logistic Regression model 
# and generate predictions.


# --- KNN ---
# Q1

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)

knn_predictions = knn_pipeline.predict(X_test)

print('\nKNN Q1: Test Accuracy:')
print(accuracy_score(y_test, knn_predictions))

print('\nKNN Q1: Classification Report:')
print(classification_report(y_test, knn_predictions))


# Q2

k_values = [1, 3, 5, 7, 9, 11, 15, 21]

cv_results = {}

print('\nKNN Q2: Mean 5-Fold Cross-Validation Accuracy:')

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

print(f'\nBest k based on CV accuracy: {best_k}')

# I would select the k value with the highest average 
# cross-validation accuracy because it demonstrated 
# the best performance across all training folds.


# Q3

# KNN without scaling
knn_raw = KNeighborsClassifier(n_neighbors=5)

knn_raw.fit(X_train, y_train)

raw_predictions = knn_raw.predict(X_test)

raw_accuracy = accuracy_score(y_test, raw_predictions)

# KNN with scaling
knn_scaled = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

knn_scaled.fit(X_train, y_train)

scaled_predictions = knn_scaled.predict(X_test)

scaled_accuracy = accuracy_score(y_test, scaled_predictions)

print('\nKNN Q3: Accuracy Comparison:')
print(f'Without scaling: {raw_accuracy:.4f}')
print(f'With scaling:    {scaled_accuracy:.4f}')


# Yes, scaling improved the model's accuracy from 92.50% 
# to 95.83%. This improvement occurs because K-Nearest 
# Neighbors (KNN) relies on distance calculations to 
# identify the closest data points. Without scaling, 
# features with larger numerical ranges can have a greater 
# influence on the results. Scaling ensures that all features 
# contribute more evenly, which can lead to better model 
# performance.


# --- Logistic Regression ---
# Q1

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)

lr_predictions = lr_pipeline.predict(X_test)

print('\nLogistic Regression Q1:')

print('Accuracy:', accuracy_score(y_test, lr_predictions))
print('Precision:', precision_score(y_test, lr_predictions))
print('Recall:', recall_score(y_test, lr_predictions))
print('F1 Score:', f1_score(y_test, lr_predictions))


# Q2

coefficients = lr_pipeline.named_steps['classifier'].coef_[0]

print('\nLogistic Regression Q2: Feature Coefficients')

for feature, coefficient in zip(numeric, coefficients):
    print(f"{feature}: {coefficient:.4f}")


# Wind speed, with a coefficient of -2.4438, and precipitation, 
# with a coefficient of -1.8891, have the strongest negative 
# influence on the model's prediction to skip running. 
# This aligns with my expectations because strong winds and 
# heavy rainfall can make outdoor running less comfortable and 
# less enjoyable.



# Q3

probabilities = lr_pipeline.predict_proba(X_test)[:, 1]

print('\nLogistic Regression Q3: First 5 Test Days')

for i, probability in enumerate(probabilities[:5], start=1):
    print(f'Day {i}: {probability:.2f}')


# The probability represents the model's estimated likelihood 
# that the weather is suitable for running. A probability close 
# to 1 indicates a higher likelihood that running conditions 
# are favorable, while a probability close to 0 indicates a 
# higher likelihood that running should be skipped. A 
# probability near 0.5 means the model is uncertain about 
# which outcome is more likely. Since this value is close 
# to 0.5, the model does not strongly favor either outcome.


# --- Evaluation ---
# Q1

cm = confusion_matrix(y_test, lr_predictions)

print('\nEvaluation Q1: Confusion Matrix')
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['skip', 'good']
)

disp.plot(cmap='Blues')

plt.title('Logistic Regression Confusion Matrix')
Path('outputs').mkdir(parents=True, exist_ok=True)
plt.savefig('outputs/logreg_confusion_matrix.png')
plt.close()

print('Confusion matrix saved to outputs/logreg_confusion_matrix.png')


# The model produced 9 false positives and 7 false negatives. 
# False positives are more concerning in a running application 
# because they could lead the app to recommend running when 
# weather conditions are unfavorable or potentially unsafe. 
# Therefore, minimizing false positives is important to ensure 
# the app provides reliable and safety-conscious recommendations.



# --- joblib ---
# Q1

joblib.dump(lr_pipeline, 'models/warmup_model.pkl')

loaded = joblib.load('models/warmup_model.pkl')

assert (loaded.predict(X_test) == lr_pipeline.predict(X_test)).all()

print('\njoblib Q1: Predictions match.')

# Saving only the Logistic Regression model would exclude the 
# StandardScaler, which is an important part of the preprocessing
# pipeline. Since the model was trained on scaled data, using 
# raw, unscaled data for predictions could lead to inaccurate 
# results. Saving the entire pipeline ensures that new data is 
# scaled consistently before being passed to the model for 
# prediction.