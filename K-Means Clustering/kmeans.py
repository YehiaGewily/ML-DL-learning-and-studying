import numpy as np
# import matplotlib.pyplot as plt 

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans:
    """
    A logical implementation of K-Means Clustering.
    
    Parameters:
    K (int): Number of clusters.
    max_iters (int): Maximum number of iterations.
    plot_steps (bool): Whether to plot steps (visualization).
    """
    def __init__(self, K: int = 5, max_iters: int = 100, plot_steps: bool = False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.X = None
        self.n_samples = None
        self.n_features = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the K-Means model to X and return cluster labels.
        
        Parameters:
        X (np.ndarray): Data samples.
        
        Returns:
        np.ndarray: Cluster labels for each sample.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize centroids randomly
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimization
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            if self.plot_steps:
                self.plot()

            # Check convergence
            if self._is_converged(centroids_old, self.centroids):
                break
            
        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def plot(self):
        """
        Visualize the clusters (Use only if matplotlib is enabled)
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

if __name__ == "__main__":
    from sklearn import datasets

    print("Running K-Means Demonstration...")
    
    np.random.seed(42)
    X, y = datasets.make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Number of clusters: {len(np.unique(y))}")

    k = KMeans(K=3, max_iters=150, plot_steps=False)
    y_pred = k.predict(X)

    # Simple accuracy check is hard for clustering without aligning labels, 
    # so we just print completion.
    print("K-Means completed successfully.")
    
    # Optional: Enable plot_steps=True in KMeans constructor to visualize if running locally with GUI
    # k.plot()
