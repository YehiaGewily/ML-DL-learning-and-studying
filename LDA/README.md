# Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is a technique used for dimensionality reduction and classification. It projects data onto a lower-dimensional space with good class-separability in order to avoid overfitting ("curse of dimensionality") and also reduce computational costs.

## Algorithm

LDA seeks to find a linear combination of features that characterizes or separates two or more classes of objects or events.

### Steps
1.  **Compute the d-dimensional mean vectors** for the different classes.
2.  **Compute the Scatter Matrices**:
    -   **Within-class scatter matrix ($S_W$)**: Measures the spread of data within the same class.
    -   **Between-class scatter matrix ($S_B$)**: Measures the distance between class means.
3.  **Compute Eigenvalues and Eigenvectors** of the matrix $S_W^{-1}S_B$.
4.  **Sort Eigenvectors** by decreasing eigenvalues.
5.  **Choose top $k$ Eigenvectors** to form a transformation matrix $W$.
6.  **Transform Data**: $Y = X \cdot W$.

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X, y)`: Computes scatter matrices and eigendecomposition.
    - `transform(X)`: Projects data onto the new subspace.

## Usage

```python
from lda import LDA
from sklearn import datasets

# Load data
data = datasets.load_iris()
X, y = data.data, data.target

# Reduce to 2 dimensions
lda = LDA(n_components=2)
lda.fit(X, y)
X_projected = lda.transform(X)
```
