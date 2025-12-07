# Support Vector Machine (SVM)

Support Vector Machine is a powerful supervised learning algorithm used for classification and regression. The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

## Algorithm

### Hinge Loss
We optimize the Hinge Loss function with L2 Regularization. The cost function $J(w, b)$ is:

$$ J(w,b) = \lambda ||w||^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i - b)) $$

where:
- $\lambda$: Regularization parameter
- $y_i$: Target label (-1 or 1)
- $w, b$: Parameters of the hyperplane

### Gradient Descent
We update the gradients based on the condition $y_i(w \cdot x_i - b) \geq 1$:

1.  **If condition met (Correctly classified and outside margin):**
    $$ w = w - \alpha (2\lambda w) $$
2.  **If condition not met (Misclassified or inside margin):**
    $$ w = w - \alpha (2\lambda w - y_i x_i) $$
    $$ b = b - \alpha (y_i) $$

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X, y)`: Trains the linear SVM using Gradient Descent.
    - `predict(X)`: Predicts classes based on the sign of equation.

## Usage

```python
from svm import SVM
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Generate data
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05)
y = np.where(y == 0, -1, 1) # Standardize labels

# Train
clf = SVM()
clf.fit(X, y)

# Predict
predictions = clf.predict(X)
```
