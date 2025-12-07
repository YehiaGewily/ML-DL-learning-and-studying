# Perceptron

The Perceptron is one of the simplest Artificial Neural Network architectures. It is a linear classifier (binary) and serves as the fundamental building block for deep learning.

## Algorithm

### Unit Step Function
The activation function used is the Unit Step Function (Heaviside Step Function):

$$
f(x) = \begin{cases} 
1 & \text{if } x \geq 0 \\
0 & \text{if } x < 0 
\end{cases}
$$

### Update Rule
For each training sample $(x, y)$:

1.  Calculate prediction: $\hat{y} = f(w \cdot x + b)$
2.  Update weights and bias:
    $$ w = w + \alpha (y - \hat{y}) x $$
    $$ b = b + \alpha (y - \hat{y}) $$
    
Where $\alpha$ is the learning rate.

## Implementation Details

- **Language**: Python
- **Key Methods**:
    - `fit(X, y)`: Trains the weights using the update rule iteratively.
    - `predict(X)`: Computes the linear combination and applies the step function.

## Usage

```python
from perceptron import Perceptron
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Generate data
X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)

# Predict
predictions = p.predict(X_test)
```
