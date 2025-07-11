import joblib
import numpy as np
import pandas as pd

print("Loading model...")
model = joblib.load("grail_model.pkl")
print("Model loaded.")

# Use DataFrame with correct column names to avoid warning
columns = ["QuizAvg", "Attendance", "Missed", "Overall"]
new_student = pd.DataFrame([[62, 68, 12, 66]], columns=columns)
print("New student data:", new_student)

prediction = model.predict(new_student)
print("Raw prediction:", prediction)

label = "At Risk" if prediction[0] == 1 else "Not At Risk"
print(f"Prediction: {label}")
