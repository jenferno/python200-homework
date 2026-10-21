# Part 2: Task 7 - Load and Predict

import json
import joblib
import pandas as pd
from pathlib import Path


model_path = Path("models/weather_classifier.pkl")
metadata_path = Path("models/weather_classifier_metadata.json")

if not model_path.exists() or not metadata_path.exists():
    raise FileNotFoundError(
        "Model files are missing. Run train_classifier.py first."
    )

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)


print("Task 7: Weather Classifier Predictions")
print("City:", metadata["city"])
print("Features:", metadata["features"])
print(f"Test Accuracy: {metadata['test_accuracy']:.2%}")

# Four hypothetical weather days
# Values: max temp (C), min temp (C), rain (mm), wind (km/h)
days = [
    [20, 10, 0, 12],    # Clearly good day
    [35, 25, 15, 45],   # Clearly bad day
    [26, 10, 2.9, 29],  # Borderline good day
    [27, 12, 3.1, 31],  # Borderline bad day
]


new_weather = pd.DataFrame(
    days,
    columns=metadata["features"]
)


predictions = model.predict(new_weather)
probabilities = model.predict_proba(new_weather)


good_class_index = list(model.classes_).index(1)

print("\nPredictions for hypothetical weather days:")

for i, (prediction, probability) in enumerate(
    zip(predictions, probabilities), start=1
):
    label = "good" if prediction == 1 else "skip"
    good_probability = probability[good_class_index]

    print(f"\nDay {i}:")
    for feature, value in new_weather.iloc[i - 1].items():
        print(f"  {feature}: {value}")

    print(f"  Prediction: {label}")
    print(f"  Probability of good: {good_probability:.2f}")


# Task 8: Reflection

# Question 1: How confident is the model when making predictions for borderline 
# weather conditions?
#
#The KNN model achieved a test accuracy of 98.63%. However, it can still make 
# incorrect predictions. For example, Day 3 was classified as unsuitable for running, 
# with a predicted probability of 0.00 for favorable running conditions, even though 
# it meets all the criteria for a good running day. This demonstrates that high overall 
# accuracy does not guarantee correct predictions for every individual case.
#
#Since my KNN model uses k=1, its probability estimates are either 0.00 or 1.00, which 
# limits its ability to communicate uncertainty. If a model assigned a borderline day 
# a probability of 0.52 for favorable running conditions, I would interpret that result 
# as uncertain rather than a confident recommendation. In that situation, I would 
# suggest checking the latest weather forecast before deciding whether to run.

#Question 2: What happens if `predict.py` runs before `train_classifier.py`?#
#
# If `predict.py` runs before `train_classifier.py`, the saved model and metadata files 
# may not exist, preventing the prediction script from loading the necessary resources. 
# To address this issue, I implemented a file-existence check that raises a 
# `FileNotFoundError` with the message: "Model files are missing. Run train_classifier.py 
# first." This provides a clear explanation of the problem and tells the user how to 
# resolve it before attempting to generate predictions.

# Question 3: What changes would be needed for a Week 10 automated cloud pipeline?
#
# To adapt this project for an automated cloud pipeline, I would replace the hypothetical 
# weather examples with current weather data retrieved from an API or cloud storage. 
# I would store the trained model in a cloud-accessible location and schedule the 
# pipeline to generate updated predictions automatically. Additionally, I would implement 
# logging and error handling to identify issues such as missing data, unavailable model 
# files, or failed API requests. These improvements would allow the system to provide 
# updated running recommendations with minimal manual intervention.