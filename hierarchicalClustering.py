import numpy as np
from scipy.spatial.distance import pdist, squareform

class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        """
        Perform hierarchical clustering using complete linkage.
        """
        # Compute pairwise distances between all points
        distances = pdist(X, 'euclidean')  # Euclidean distance
        distance_matrix = squareform(distances)  # Convert to square form

        # Create a cluster for each point initially (each point is its own cluster)
        clusters = [[i] for i in range(len(X))]

        # Perform agglomerative clustering
        while len(clusters) > self.n_clusters:
            # Find the two closest clusters
            min_dist = np.inf
            pair_to_merge = None
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    # Compute the complete linkage distance (maximum distance between points)
                    cluster_i = clusters[i]
                    cluster_j = clusters[j]
                    max_dist = 0
                    for idx_i in cluster_i:
                        for idx_j in cluster_j:
                            max_dist = max(max_dist, distance_matrix[idx_i][idx_j])
                    if max_dist < min_dist:
                        min_dist = max_dist
                        pair_to_merge = (i, j)

            # Merge the two closest clusters
            i, j = pair_to_merge
            new_cluster = clusters[i] + clusters[j]
            clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]
            clusters.append(new_cluster)

        # Assign labels to each data point
        self.labels_ = np.zeros(len(X), dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = label

    def predict(self, X):
        """
        Predict the labels of the clusters (this is available after fitting the model).
        """
        return self.labels_

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate synthetic dataset
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    # Fit hierarchical clustering model
    model = HierarchicalClustering(n_clusters=3)
    model.fit(X)

    # Get the labels (cluster assignments)
    labels = model.predict(X)

    # Plot the clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("Hierarchical Clustering with Complete Linkage")
    plt.show()
