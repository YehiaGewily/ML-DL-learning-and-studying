import numpy as np

class LogisticRegression:
    """
    A logical implementation of Logistic Regression for Binary Classification.
    
    Parameters:
    lr (float): Learning rate (alpha).
    n_iters (int): Number of iterations for gradient descent.
    """
    def __init__(self, lr: float = 0.01, n_iters: int = 10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the logistic regression model to the training data.

        Parameters:
        X (np.ndarray): Training features of shape (n_samples, n_features).
        y (np.ndarray): Training labels of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Optional: Print cost occasionally
            if i % 1000 == 0:
                cost = self._compute_cost(y, y_predicted)
                print(f"Iteration {i}: Cost {cost:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters:
        X (np.ndarray): Samples to predict, of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted class labels (0 or 1) of shape (n_samples,).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _compute_cost(self, y_true, y_pred):
        # Avoid division by zero
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    print("Running Logistic Regression Demonstration...")
    
    # Generate synthetic binary data
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=2, random_state=1)
    
    # Model training
    model = LogisticRegression(lr=0.1, n_iters=5000)
    model.fit(X, y)
    
    # Prediction
    predictions = model.predict(X)
    
    accuracy = np.mean(y == predictions)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
