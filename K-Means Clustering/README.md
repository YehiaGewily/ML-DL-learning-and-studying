# K-Means Clustering

K-Means is an unsupervised learning algorithm that partitions a dataset into $K$ pre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group.

## Algorithm

### Steps
1.  **Initialization**: Randomly select $K$ data points as initial centroids.
2.  **Assignment**: Assign each data point to the closest centroid (using Euclidean distance).
3.  **Update**: Calculate new centroids by taking the mean of all data points in each cluster.
4.  **Repeat**: Repeat steps 2 and 3 until convergence (centroids do not change).

### Complexity
-   Time Complexity: $O(I \cdot K \cdot N \cdot d)$ where $I$ is iterations, $N$ is samples, $d$ is dimensions.

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `predict(X)`: Runs the optimization loop and assigns labels.
    - `plot()`: Visualizes the clustering steps (requires Matplotlib).

## Usage

```python
from kmeans import KMeans
from sklearn import datasets

# Generate data
X, y = datasets.make_blobs(centers=3, n_samples=500, n_features=2)

# Train and Predict
k = KMeans(K=3, max_iters=100)
y_pred = k.predict(X)
```
