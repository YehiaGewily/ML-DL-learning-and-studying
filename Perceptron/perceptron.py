import numpy as np

class Perceptron:
    """
    A logical implementation of the Perceptron algorithm, the fundamental building block of Neural Networks.
    
    Parameters:
    learning_rate (float): Learning rate (alpha).
    n_iters (int): Number of iterations over the training set.
    """
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Perceptron model to the training data.
        
        Parameters:
        X (np.ndarray): Training features.
        y (np.ndarray): Training labels (0 or 1).
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Ensure labels are binary (0, 1) if not already
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                # w = w + alpha * (y - y_hat) * x
                update = self.lr * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Returns:
        np.ndarray: Predicted labels (0 or 1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print("Running Perceptron Demonstration...")
    
    # Generate synthetic linearly separable data
    X, y = datasets.make_blobs(
        n_samples=1000, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    
    y_pred = p.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
