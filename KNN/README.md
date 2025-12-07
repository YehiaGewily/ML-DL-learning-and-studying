# K-Nearest Neighbors (KNN)

KNN is a non-parametric, lazy learning algorithm used for classification and regression. It makes predictions based on the $k$ closest training examples in the feature space.

## Algorithm

### Euclidean Distance
To determine which points are closest, we calculate the Euclidean distance between the new point $x$ and existing points $x^{(i)}$.

$$ d(x, x^{(i)}) = \sqrt{\sum_{j=1}^{n} (x_j - x^{(i)}_j)^2} $$

### Voting
1.  **Calculate Distances**: Compute distance between the query point and all training samples.
2.  **Find Neighbors**: Sort the distances and select the top $k$ nearest samples.
3.  **Vote**:
    -   **Classification**: The class is determined by the majority vote of the neighbors.
    -   **Regression**: The value is the average of the neighbors' values.

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X, y)`: Stores training data (Lazy learner).
    - `predict(X)`: Computes distances and performs voting.
- **Optimization**: Uses vectorized operations for efficient distance calculation.

## Usage

```python
from knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Generate data
X, y = datasets.make_classification(n_samples=100, n_features=5, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
knn = KNN(k=3)
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)
```
