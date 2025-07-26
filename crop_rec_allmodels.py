import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 1. Load dataset
df = pd.read_csv("D:/E/2nd SEM/project EX/crop recomondetion/Crop_recommendation.csv")  # Replace with your actual CSV file name

# 2. Split features and label
X = df.drop('label', axis=1)
y = df['label']

# 3. Encode label (target column)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB()
}

# 7. Evaluate all models
results = []

for name, model in models.items():
    print(f"Training {name}...")

    start_time = time.time()

    # Use scaled features for specific models
    if name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine", "Naive Bayes"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    end_time = time.time()

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    error = 1 - acc
    train_time = round(end_time - start_time, 4)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'F1 Score': round(f1, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'Error Rate': round(error, 4),
        'Training Time (s)': train_time,
        'Confusion Matrix': cm.tolist()
    })

# 8. Results as DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy', ascending=False)

print("\n=== Model Comparison Results ===")
print(results_df[['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'Error Rate', 'Training Time (s)']])

#csv=results_df.to_csv("crop_rec_models.csv")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict with your trained model (example: Random Forest)
y_pred = model.predict(X_test)  # Or X_test_scaled if needed

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# Plot with title and customization
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax, xticks_rotation=90)
plt.title("Confusion Matrix - Random Forest")
plt.grid(False)
plt.tight_layout()
plt.show()

####
for name, model in models.items():
    if name in ["Logistic Regression", "K-Nearest Neighbors", "SVM", "Naive Bayes"]:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=90)
    plt.title(f"Confusion Matrix - {name}")
    plt.grid(False)
    plt.tight_layout()
    plt.show()



