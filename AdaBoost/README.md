# AdaBoost (Adaptive Boosting)

AdaBoost is a popular boosting technique which helps you combine multiple "weak classifiers" into a single "strong classifier". A weak classifier is simply a classifier that performs poorly, but performs better than random guessing.

## Algorithm

### Weak Classifiers (Decision Stumps)
In this implementation, we use **Decision Stumps** as weak classifiers. A decision stump is a decision tree with a depth of 1 (i.e., it splits the data based on a single feature and a single threshold).

### Weights
1.  **Initialize Weights**: $w_i = 1/N$ for all samples.
2.  **Train Weak Classifier**: Find the best stump that minimizes the weighted classification error:
    $$ \epsilon = \sum w_i \mathds{1}(y_i \neq h(x_i)) $$
3.  **Calculate Say (Alpha)**:
    $$ \alpha = \frac{1}{2} \ln \left( \frac{1-\epsilon}{\epsilon} \right) $$
4.  **Update Weights**:
    $$ w_i \leftarrow w_i \cdot \exp(-\alpha y_i h(x_i)) $$
    $$ w_i \leftarrow \frac{w_i}{\sum w_j} $$ (Normalization)

### Final Prediction
The final prediction is a weighted sum of the weak classifiers:
$$ H(x) = \text{sign} \left( \sum_{t=1}^{T} \alpha_t h_t(x) \right) $$

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X, y)`: Iteratively trains stumps and updates sample weights.
    - `predict(X)`: Aggregates predictions.

## Usage

```python
from adaboost import AdaBoost
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
data = datasets.load_breast_cancer()
X, y = data.data, data.target
y = np.where(y == 0, -1, 1) # Must use -1/1 labels

# Train
clf = AdaBoost(n_clf=5)
clf.fit(X, y)

# Predict
predictions = clf.predict(X)
```
