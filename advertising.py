import pandas as pd
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For train/test split
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Performance metrics

# Load the CSV file into a DataFrame
data = pd.read_csv("Advertising.csv")  # Ensure the correct path is specified

# Display the first few rows of the DataFrame
data.head()

# Define feature columns and target variable
features = ["TV", "radio", "newspaper"]
X = data[features]  # Extract feature columns
y = data["sales"]  # Extract target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

# Add intercept (bias term) to feature matrices
X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

# Convert target variables to NumPy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Calculate regression coefficients using the Normal Equation
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predict values for both training and testing sets
y_train_pred = X_train @ beta
y_test_pred = X_test @ beta

# Calculate performance metrics
RSS = np.sum((y_train - y_train_pred) ** 2)
TSS = np.sum((y_train - y_train.mean()) ** 2)
R2 = 1 - RSS / TSS
RSE = np.sqrt(RSS / (X_train.shape[0] - X_train.shape[1]))

# Additional performance metrics
MSE = mean_squared_error(y_test, y_test_pred)
MAE = mean_absolute_error(y_test, y_test_pred)

# Display calculated metrics and results
print(f"Coefficients: {beta}")
print(f"RSS: {RSS:.4f}")
print(f"R²: {R2:.4f}")
print(f"RSE: {RSE:.4f}")
print(f"MSE: {MSE:.4f}")
print(f"MAE: {MAE:.4f}")

# Visualizing predicted vs actual sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit Line')
plt.title("Predicted vs Actual Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.legend()
plt.grid(True)
plt.show()
