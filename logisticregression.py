import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample Dataset representing exam scores and admission status
data = {
    'Exam1': [34, 45, 50, 61, 70, 80, 85, 95, 30, 60],  # Exam 1 scores
    'Exam2': [78, 56, 78, 82, 60, 90, 40, 95, 30, 50],  # Exam 2 scores
    'Admitted': [0, 0, 1, 1, 1, 1, 0, 1, 0, 1]          # Admission status (0 = No, 1 = Yes)
}
df = pd.DataFrame(data)

# Features (Exam1 and Exam2 scores) and Target (Admitted status)
X = df[['Exam1', 'Exam2']]  # Independent variables
y = df['Admitted']          # Dependent variable

# Splitting data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()  # Logistic regression classifier
model.fit(X_train, y_train)   # Train the model using the training data

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluation of model performance
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")  # Display accuracy score
print("Confusion Matrix:")  # Display confusion matrix for true vs predicted labels
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")  # Display precision, recall, and F1 score
print(classification_report(y_test, y_pred))
