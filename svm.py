import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Support Vector Machine (SVM) classifier using Stochastic Gradient Descent.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        """
        Train the SVM model using the provided training data.
        """
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        """
        Make predictions using the trained SVM model.
        """
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)

# Example usage with Iris dataset:
if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Convert to binary classification (Setosa vs Non-Setosa)
    y = np.where(y == 0, -1, 1)  # Setosa is -1, others are 1
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM model
    model = SVM()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
