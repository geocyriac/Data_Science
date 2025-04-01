import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2):
        """
        Initialize the RandomForest model with the specified number of trees,
        maximum depth, and minimum samples required to split a node.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []  # List to store individual decision trees
        
    def fit(self, X, y):
        """
        Train the random forest model using bootstrap sampling.
        Each tree is trained on a randomly resampled dataset.
        """
        self.trees = []
        for _ in range(self.n_trees):
            # Create a bootstrap sample of the dataset
            X_sample, y_sample = resample(X, y)
            
            # Train a decision tree on the sampled data
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            
            # Store the trained tree
            self.trees.append(tree)
            
    def predict(self, X):
        """
        Predict the target values for the given input data.
        The final prediction is obtained by averaging the predictions from all trees.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])  # Collect predictions from each tree
        return np.mean(predictions, axis=0)  # Compute the average prediction

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    # Generate synthetic regression dataset
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the RandomForest model
    model = RandomForest(n_trees=10, max_depth=5)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate model performance using Mean Squared Error (MSE)
    print("MSE:", mean_squared_error(y_test, y_pred))
