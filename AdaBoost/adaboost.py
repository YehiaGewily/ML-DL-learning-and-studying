import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        
        return predictions


class AdaBoost:
    """
    A logical implementation of AdaBoost (Adaptive Boosting) using Decision Stumps.
    
    Parameters:
    n_clf (int): Number of weak classifiers (decision stumps).
    """
    def __init__(self, n_clf: int = 5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the AdaBoost model to the training data.
        
        Parameters:
        X (np.ndarray): Training features.
        y (np.ndarray): Training labels (-1 or 1).
        """
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            # Greedy search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # Check both polarities (1: < threshold is -1, otherwise 1)
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error is sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i

            # Calculate alpha (say or vote power)
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + EPS))
            
            # Calculate predictions to update weights
            predictions = clf.predict(X) # Make sure to use the predict logic of the stump

            # Update weights: increase weight of misclassified
            # w = w * exp(-alpha * y * pred)
            # if y == pred, y*pred = 1 -> exp(-alpha) -> weight decreases
            # if y != pred, y*pred = -1 -> exp(alpha) -> weight increases
            w *= np.exp(-clf.alpha * y * predictions)
            
            # Normalize weights
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Returns:
        np.ndarray: Predicted labels (-1 or 1).
        """
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        return np.sign(y_pred)


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    print("Running AdaBoost Demonstration...")
    
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    
    # Convert labels to -1, 1
    y = np.where(y == 0, -1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    clf = AdaBoost(n_clf=5)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
