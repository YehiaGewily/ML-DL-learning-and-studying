# Naive Bayes

Naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

## Algorithm

### Bayes' Theorem
$$ P(y|X) = \frac{P(X|y) P(y)}{P(X)} $$

Where:
- $P(y|X)$ is the posterior probability of class $y$ given predictor $X$.
- $P(y)$ is the prior probability of class $y$.
- $P(X|y)$ is the likelihood which is the probability of predictor $X$ given class $y$.
- $P(X)$ is the prior probability of predictor $X$.

### Gaussian Naive Bayes
This implementation assumes that the continuous values associated with each class are distributed according to a Gaussian distribution.

$$ P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right) $$

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X, y)`: Calculates mean, variance, and priors for each class.
    - `predict(X)`: Calculates posterior probabilities and selects the class with max probability.

## Usage

```python
from naive_bayes import NaiveBayes
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Generate data
X, y = datasets.make_classification(n_samples=100, n_features=10, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Predict
predictions = nb.predict(X_test)
```
