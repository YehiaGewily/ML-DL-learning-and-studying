import numpy as np

class KNN:
    """
    A logical implementation of K-Nearest Neighbors Classifier.
    
    Parameters:
    k (int): Number of neighbors to use.
    """
    def __init__(self, k: int = 3):
        self.k = k
        self.eps = 1e-8
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Store the training data.

        Parameters:
        X (np.ndarray): Training features of shape (n_samples, n_features).
        y (np.ndarray): Training labels of shape (n_samples,).
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters:
        X (np.ndarray): Samples to predict, of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted class labels.
        """
        distances = self._compute_distance_vectorized(X)
        return self._predict_labels(distances)

    def _compute_distance_vectorized(self, X_test):
        # (X_test - X_train)^2 = X_test^2 - 2*X_test*X_train + X_train^2
        X_test_squared = np.sum(X_test**2, keepdims=True, axis=1)
        X_train_squared = np.sum(self.X_train**2, keepdims=True, axis=1)
        
        # Dot product
        X_test_train_dot = np.dot(X_test, self.X_train.T)
        
        # Euclidean distance
        # Added epsilon for numerical stability
        distances = np.sqrt(
            self.eps + X_test_squared - 2 * X_test_train_dot + X_train_squared.T
        )
        return distances

    def _predict_labels(self, distances):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            # Sort by distance and get indices of k nearest neighbors
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.y_train[y_indices[:self.k]].astype(int)
            
            # Majority vote
            # bincount gives the count of each non-negative integer in array
            # argmax gives the value with max count
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))

        return y_pred


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print("Running KNN Demonstration...")
    
    # Generate synthetic data
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=3, n_informative=5, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)

    print(f"Accuracy: {accuracy * 100:.2f}%")
