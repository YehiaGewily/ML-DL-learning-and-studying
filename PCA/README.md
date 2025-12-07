# Principal Component Analysis (PCA)

PCA is an unsupervised learning technique used for dimensionality reduction. It identifies a set of orthogonal axes (principal components) that capture the maximum variance in the data.

## Algorithm

### Steps
1.  **Mean Centering**: Subtract the mean of each feature from the dataset.
2.  **Covariance Matrix**: Calculate the covariance matrix of the features.
    $$ Cov(X) = \frac{1}{n-1} X^T X $$
3.  **Eigendecomposition**: Calculate eigenvalues and eigenvectors of the covariance matrix.
4.  **Sort**: Sort eigenvectors by decreasing eigenvalues.
5.  **Projection**: Choose top $k$ eigenvectors ($W$) and project the data.
    $$ Y = X \cdot W $$

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X)`: Computes mean, covariance, and eigendecomposition.
    - `transform(X)`: Projects data onto the principal components.

## Usage

```python
from pca import PCA
from sklearn import datasets

# Load data
data = datasets.load_iris()
X = data.data

# Reduce to 2 dimensions
pca = PCA(n_components=2)
pca.fit(X)
X_projected = pca.transform(X)
```
