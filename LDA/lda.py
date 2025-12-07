import numpy as np

class LDA:
    """
    A logical implementation of Linear Discriminant Analysis (LDA) for dimensionality reduction.
    
    Parameters:
    n_components (int): Number of components (features) to keep.
    """
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the LDA model to the training data.
        
        Parameters:
        X (np.ndarray): Training features of shape (n_samples, n_features).
        y (np.ndarray): Training labels.
        """
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Mean of the entire dataset
        mean_overall = np.mean(X, axis=0)
        
        S_W = np.zeros((n_features, n_features)) # Within-class scatter matrix
        S_B = np.zeros((n_features, n_features)) # Between-class scatter matrix
        
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            
            # (n_c, n_features) -> (n_features, n_features)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
        
        # Determine SW^-1 * SB
        A = np.linalg.inv(S_W).dot(S_B)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Transpose eigenvectors for easier row-wise handling
        eigenvectors = eigenvectors.T 
        
        # Sort eigenvalues in descending order
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # Store first n_components eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the linear discriminants.
        
        Parameters:
        X (np.ndarray): Data to transform.
        
        Returns:
        np.ndarray: Transformed data with reduced dimensions.
        """
        return np.dot(X, self.linear_discriminants.T)


if __name__ == "__main__":
    from sklearn import datasets
    # import matplotlib.pyplot as plt # Optional for visualization

    print("Running LDA Demonstration...")
    
    # Load dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target
    
    # Fit LDA
    lda = LDA(2)
    lda.fit(X, y)
    
    # Transform
    X_projected = lda.transform(X)
    
    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)
