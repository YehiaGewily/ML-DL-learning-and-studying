# Decision Tree

A Decision Tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).

## Algorithm

This implementation uses the ID3/CART algorithms concept, utilizing **Information Gain** to decide where to split.

### Entropy
Entropy measures the impurity of the data. High entropy means the data is mixed, while low entropy means the data is pure.

$$ E(S) = - \sum_{i=1}^{c} p_i \log_2(p_i) $$

### Information Gain
Information gain calculates the reduction in entropy from transforming the dataset in some way. We choose the split that maximizes information gain.

$$ IG(S, A) = E(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} E(S_v) $$

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X, y)`: Builds the tree recursively.
    - `predict(X)`: Traverses the tree for new samples.

## Usage

```python
from decision_tree import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load data
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
```
