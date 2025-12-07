import numpy as np

class SVM:
    """
    A logical implementation of Linear SVM using Hinge Loss and Gradient Descent.
    
    Parameters:
    learning_rate (float): Learning rate for gradient descent.
    lambda_param (float): Regularization parameter.
    n_iters (int): Number of iterations.
    """
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVM model to the training data.
        
        Parameters:
        X (np.ndarray): Training features of shape (n_samples, n_features).
        y (np.ndarray): Training labels (must be -1 or 1).
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Ensure labels are -1 and 1
        y_ = np.where(y <= 0, -1, 1) # converts 0/1 to -1/1 if needed

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Returns:
        np.ndarray: Predicted labels (-1 or 1, or mapped back to 0/1 if preferred).
        """
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print("Running SVM Demonstration...")
    
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    # Convert y to -1, 1 for consistent checking
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    clf = SVM()
    clf.fit(X_train, y_train)
    
    predictions = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.sum(predictions == y_test) / len(y_test)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
