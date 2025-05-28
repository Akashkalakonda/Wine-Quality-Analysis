# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import joblib  # Library to save the model

# Load the dataset
# Replace 'winequality.csv' with your file path if needed
data = pd.read_csv('WINEDATASET.csv')


# Categorize wine quality
def categorize_quality(quality):
    if quality < 5:
        return "low"
    elif 5 <= quality <= 6:
        return "medium"
    else:
        return "high"

# Apply the function to the quality column
data["quality_category"] = data["quality"].apply(categorize_quality)

# Drop the original quality column
data = data.drop("quality", axis=1)

# Separate features (X) and target (y)
X = data.drop("quality_category", axis=1)  # Input features
y = data["quality_category"]              # Target variable

# Encode the target variable
y = y.map({"low": 0, "medium": 1, "high": 2})  # Map categories to integers for classification

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Random Forest Classifier Results:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Optional: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["low", "medium", "high"]))

# Save the model
model_filename = "RANDOM_FOREST_WINE_CLASSIFIER.pkl"
joblib.dump(model, model_filename)

print(f"\nModel saved successfully as {model_filename}")
