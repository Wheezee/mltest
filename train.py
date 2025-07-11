import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("grail_dummy_data_100.csv")

# Encode target label (Yes/No → 1/0)
le = LabelEncoder()
df["AtRisk"] = le.fit_transform(df["AtRisk"])  # Yes=1, No=0

# Features and label
X = df[["QuizAvg", "Attendance", "Missed", "Overall"]]
y = df["AtRisk"]

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Not At Risk", "At Risk"]))

# Save the model
joblib.dump(model, "grail_model.pkl")
print("\nModel saved as grail_model.pkl")
