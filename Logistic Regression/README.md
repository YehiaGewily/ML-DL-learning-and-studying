# Logistic Regression

Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

## Algorithm

Logistic regression differs from linear regression by using the sigmoid function to map predictions to probabilities between 0 and 1, making it suitable for classification tasks.

### Sigmoid Function
The hypothesis function uses the sigmoid (or logistic) function:

$$ g(z) = \frac{1}{1 + e^{-z}} $$

where $z = \theta^T x$. Thus, $h_\theta(x) = g(\theta^T x)$.

### Cost Function (Log Loss)
We use the Cross-Entropy Loss (or Log Loss) as the cost function:

$$ J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))] $$

### Gradient Descent
The update rule typically looks identical to linear regression, but the definition of $h_\theta(x)$ is different.

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$

## Implementation Details

- **Language**: Python
- **Libraries**: NumPy
- **Key Methods**:
    - `fit(X, y)`: Trains the model.
    - `predict(X)`: Predicts binary class labels (0/1).

## Usage

```python
from logistic_regression import LogisticRegression
from sklearn.datasets import make_blobs

# Generate data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Train model
clf = LogisticRegression(lr=0.1, n_iters=1000)
clf.fit(X, y)

# Predict
predictions = clf.predict(X)
```
