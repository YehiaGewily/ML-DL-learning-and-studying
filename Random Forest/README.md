# Random Forest

Random Forest is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

## Algorithm

### Bootstrapping
We train multiple decision trees on different subsets of the dataset. These subsets are created by random sampling with replacement (bootstrapping).

### Feature Randomness
In a normal decision tree, when splitting a node, we look at every possible feature. In Random Forest, at each split, we randomly select a subset of features and find the best split among them. This prevents the trees from being too correlated.

### Aggregation (Bagging)
The final prediction is made by aggregating the predictions of all individual trees. For classification, this is a majority vote.

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X, y)`: Trains `n_trees` Decision Trees.
    - `predict(X)`: Aggregates votes from all trees.

## Usage

```python
from random_forest import RandomForest
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load data
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
```
