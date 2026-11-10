import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    cnt = 0
    bs = 1
    w = np.array(weights, dtype=float).copy()
    if method == "stochastic":
        bs = 1
    elif method == "batch":
        bs = len(X)
    elif method == "mini_batch":
        bs = batch_size
    X = np.array(X)
    y = np.array(y)
    for epoch in range(n_epochs):
        cnt = 0
        while cnt*bs < X.shape[0]:
            X_batch = X[cnt * bs:(cnt+1) * bs, :]
            y_batch = y[cnt * bs:(cnt+1) * bs]
            grad = 2*X_batch.T @ (X_batch @ w -y_batch)
            grad /= X_batch.shape[0]
            w -= grad * learning_rate
            cnt += 1

    return w
