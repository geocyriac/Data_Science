import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Sample Data (2x2 matrix)
X = np.array([[2, 3],
              [4, 5],
              [6, 7],
              [8, 9]])

# Class labels
y = np.array([0, 0, 1, 1])  # Binary classification

# ---------------------- Using Library ----------------------
lda = LDA()
X_lda_lib = lda.fit_transform(X, y)
print("LDA Transformed Data (Using Library):\n", X_lda_lib)

# ---------------------- Manual Calculation ----------------------
# Calculate means for each class
mean_0 = np.mean(X[y == 0], axis=0)  # Mean of class 0
mean_1 = np.mean(X[y == 1], axis=0)  # Mean of class 1

# Calculate overall mean
mean_total = np.mean(X, axis=0)

# Compute Scatter matrices
S_W = np.cov(X[y == 0].T) + np.cov(X[y == 1].T)  # Within-class scatter matrix
S_B = np.outer(mean_0 - mean_total, mean_0 - mean_total) + \
      np.outer(mean_1 - mean_total, mean_1 - mean_total)  # Between-class scatter matrix

# Regularization to prevent singular matrix error
epsilon = 1e-6  # Small constant for numerical stability
S_W += epsilon * np.eye(S_W.shape[0])

# Compute Eigenvalues and Eigenvectors
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_W) @ S_B)

# Sort eigenvectors by eigenvalues in descending order
eigvecs = eigvecs[:, np.argsort(-eigvals)]

# Project data
X_lda_manual = X @ eigvecs[:, 0].reshape(-1, 1)  # Project data on top eigenvector

print("LDA Transformed Data (Manual Calculation):\n", X_lda_manual)
