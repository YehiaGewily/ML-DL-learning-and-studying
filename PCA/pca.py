import numpy as np

class PCA:
    """
    A logical implementation of Principal Component Analysis (PCA) for dimensionality reduction.
    
    Parameters:
    n_components (int): Number of principal components to keep.
    """
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the PCA model to the training data.
        
        Parameters:
        X (np.ndarray): Training features of shape (n_samples, n_features).
        """
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        # row = features, columns = samples (needs transpose of X where rows are samples)
        # np.cov expects rows as variables (features) and columns as observations
        cov = np.cov(X_centered.T)

        # Eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Transpose eigenvectors for easier row-wise handling
        eigenvectors = eigenvectors.T
        
        # Sort eigenvectors by eigenvalues in descending order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # Store first n_components eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the principal components.
        
        Parameters:
        X (np.ndarray): Data to transform.
        
        Returns:
        np.ndarray: Transformed data with reduced dimensions.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)


if __name__ == "__main__":
    from sklearn import datasets
    # import matplotlib.pyplot as plt

    print("Running PCA Demonstration...")
    
    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(n_components=2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Shape of original X:', X.shape)
    print('Shape of transformed X:', X_projected.shape)
    
    # Optional plotting code
    # x1 = X_projected[:,0]
    # x2 = X_projected[:,1]
    # plt.scatter(x1, x2, c=y, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.colorbar()
    # plt.show()
