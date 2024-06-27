
# Simple Neural Network for MNIST from Scratch using NumPy

This project implements a simple neural network for classifying handwritten digits from the MNIST dataset. The neural network is built from scratch using only NumPy, demonstrating fundamental concepts of neural networks including forward propagation, backward propagation, and gradient descent.


## Features

- Forward Propagation: Implemented for a two-layer neural network.
- Backward Propagation: Includes the calculation of gradients for weights and biases.
- Gradient Descent: Updates weights and biases to minimize the loss function.
- MNIST Data Handling: Loading and preprocessing of the MNIST dataset.
-Visualization: Displays sample predictions with the corresponding handwritten digit images.


## Requirements
- Python 3.x
- NumPy
- pandas
- Matplotlib
## Installation

Clone the repository:

```bash
  git clone https://github.com/yourusername/mnist-numpy-neural-network.git
  cd mnist-numpy-neural-network
```

Install dependencies:
```bash
pip install numpy matplotlib
pip install pandas
pip install numpy
```
Download the MNIST dataset:
```bash
You can download the dataset from [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) or use the provided script.
```
## Training the Neural Network

```javascript
Initialize Parameters:

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
```
```javascript
Forward Propagation:

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```
```javascript
Backward Propagation:

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
```
```javascript
Update Parameters:

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
```
```javascript
Gradient Descent:

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
```
## Training the Neural Network
```javascript
Make Predictions:

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
```
```javascript
Test Prediction::
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
```

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). See the LICENSE file for details.

