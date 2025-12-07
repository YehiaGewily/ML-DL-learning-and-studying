import numpy as np


class LinearRegression:
    """
    A logical implementation of Linear Regression using Gradient Descent.
    
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
        Fit the linear regression model to the training data.

        Parameters:
        X (np.ndarray): Training features of shape (n_samples, n_features).
        y (np.ndarray): Training labels/targets of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Optional: Print loss occasionally
            if i % 1000 == 0:
                loss = self._mean_squared_error(y, y_predicted)
                print(f"Iteration {i}: Loss {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given features.

        Parameters:
        X (np.ndarray): Samples to predict, of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted values of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias

    def _mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    # Demonstration
    print("Running Linear Regression Demonstration...")
    
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1) * 0.5
    
    # Flatten y for easier handling (n_samples,)
    y = y.flatten()

    # Model training
    regressor = LinearRegression(lr=0.01, n_iters=10000)
    regressor.fit(X, y)
    
    # Prediction
    predictions = regressor.predict(X)
    
    print(f"True weights: 3, True bias: 4")
    print(f"Learned weights: {regressor.weights[0]:.4f}, Learned bias: {regressor.bias:.4f}")
