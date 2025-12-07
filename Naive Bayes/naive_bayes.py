import numpy as np

class NaiveBayes:
    """
    A logical implementation of Gaussian Naive Bayes Classifier.
    """
    def __init__(self):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}
        self.num_classes = None
        self.eps = 1e-6

    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model to the training data.
        
        Parameters:
        X (np.ndarray): Training features of shape (n_samples, n_features).
        y (np.ndarray): Training labels of shape (n_samples,).
        """
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))

        for c in range(self.num_classes):
            X_c = X[y == c]
            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / self.num_examples

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (np.ndarray): Samples to predict, of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted class labels.
        """
        probs = np.zeros((X.shape[0], self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self._density_function(
                X, self.classes_mean[str(c)], self.classes_variance[str(c)]
            )
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def _density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        # Using Log prob to avoid underflow
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print("Running Naive Bayes Demonstration...")
    
    # Generate synthetic data
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    NB = NaiveBayes()
    NB.fit(X_train, y_train)
    
    y_pred = NB.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)

    print(f"Accuracy: {accuracy * 100:.2f}%")
