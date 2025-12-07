# Linear Regression

Linear Regression is a linear approach for modeling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables).

## Algorithm

The goal of linear regression is to find the best-fitting line (or hyperplane) through a set of points. The "best" line is defined as the one that minimizes the sum of squared differences (residuals) between the observed values and the values predicted by the linear model.

### Hypothesis Function
The hypothesis function $h_\theta(x)$ is given by:

$$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n $$

In vectorized form:

$$ h_\theta(x) = \theta^T x $$

### Cost Function (Mean Squared Error)
To measure the accuracy of our hypothesis, we use a cost function $J(\theta)$. The most common choice is the Mean Squared Error (MSE):

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$

### Gradient Descent
To minimize the cost function $J(\theta)$, we use Gradient Descent. We update the parameters $\theta$ simultaneously:

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$

Where $\alpha$ is the learning rate.

## Implementation Details

- **Language**: Python
- **Libraries**: NumPy
- **Key Methods**:
    - `fit(X, y)`: Trains the model using gradient descent.
    - `predict(X)`: Predicts output for new data.

## Usage

```python
import numpy as np
from linear_regression import LinearRegression

# Data preparation
X = np.random.rand(100, 1)
y = 3 * X.flatten() + 4 + np.random.randn(100) * 0.1

# Initialize and train
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X, y)

# Predict
predictions = model.predict(X)
```
