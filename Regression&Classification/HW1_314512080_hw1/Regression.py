import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(layer_dims):
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

    return parameters

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear(Z):
    A = Z
    return A, Z

def linear_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    return dZ

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation_fn):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation_fn(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_propagation(X, parameters, layer_activations):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f'W{l}'], parameters[f'b{l}'], layer_activations[l-1])
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters[f'W{L}'], parameters[f'b{L}'], layer_activations[L-1])
    caches.append(cache)
    
    return AL, caches

def compute_mse_loss(AL, Y):
    m = Y.shape[1]
    loss = (1/m) * np.sum((AL - Y)**2)
    return loss

def compute_cross_entropy_loss(AL, Y):
    m = Y.shape[1]
    loss = - (1/m) * np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8))
    loss = np.squeeze(loss)
    return loss

def linear_module_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation_fn_backward):
    linear_cache, activation_cache = cache
    dZ = activation_fn_backward(dA, activation_cache)
    dA_prev, dW, db = linear_module_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward_propagation(AL, Y, caches, layer_activations_backward, task_type="regression"):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    if task_type == "regression":
        dAL = - (Y - AL)
    elif task_type == "classification":
        dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))

    current_cache = caches[L-1]
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(dAL, current_cache, layer_activations_backward[L-1])

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f'dA{l+1}'], current_cache, layer_activations_backward[l])
        grads[f'dA{l}'] = dA_prev_temp
        grads[f'dW{l+1}'] = dW_temp
        grads[f'db{l+1}'] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    return parameters


def load_and_preprocess_data(file_path, target_column, test_split_ratio=0.2):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return None, None, None, None
    
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)


    all_target_columns = ['Heating Load', 'Cooling Load'] # load 不當作特徵
    
    X = data.drop(columns=all_target_columns)
    
    y = data[[target_column]]

    split_index = int(len(data) * (1 - test_split_ratio))
    X_train_raw, X_test_raw = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    train_mean = X_train_raw.mean()
    train_std = X_train_raw.std()

    X_train = (X_train_raw - train_mean) / train_std
    X_test = (X_test_raw - train_mean) / train_std
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, X_test, y_test, layer_dims, epochs, learning_rate):
    """
    訓練模型主函數
    """
    parameters = initialize_parameters(layer_dims)
    layer_activations = [relu] * (len(layer_dims) - 2) + [linear]
    layer_activations_backward = [relu_backward] * (len(layer_dims) - 2) + [linear_backward]
    
    train_loss_history = []
    test_loss_history = []

    print(f"Starting training for {epochs} epochs with learning rate {learning_rate}...")
    print(f"Network Architecture: {layer_dims}")

    for i in range(epochs):
        AL, caches = forward_propagation(X_train.T, parameters, layer_activations)
        grads = backward_propagation(AL, y_train.T, caches, layer_activations_backward, task_type="regression")
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if i % 1000 == 0 or i == epochs - 1:
            train_loss = compute_mse_loss(AL, y_train.T)
            train_loss_history.append(train_loss)
            
            AL_test, _ = forward_propagation(X_test.T, parameters, layer_activations)
            test_loss = compute_mse_loss(AL_test, y_test.T)
            test_loss_history.append(test_loss)
            
            print(f"Epoch {i}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
    return parameters, train_loss_history, test_loss_history

def predict(X, parameters, layer_dims):
    layer_activations = [relu] * (len(layer_dims) - 2) + [linear]
    y_pred, _ = forward_propagation(X.T, parameters, layer_activations)
    return y_pred.T


if __name__ == "__main__":
    
    DATASET_PATH = "2025_energy_efficiency_data.csv"
    TARGET_COLUMN = "Heating Load"
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data(
        file_path=DATASET_PATH,
        target_column=TARGET_COLUMN,
        test_split_ratio=0.25 # 7525
    )
    if X_train is not None:
        layer_dims = [X_train.shape[1], 100, 1] 
        EPOCHS = 20000
        LEARNING_RATE = 0.001
        
        trained_parameters, train_loss_history, test_loss_history = train_model(
            X_train, y_train, X_test, y_test, layer_dims, EPOCHS, LEARNING_RATE
        )

    plt.figure(figsize=(10, 6))
    log_interval = 1000
    epochs_range = np.arange(len(train_loss_history)) * log_interval
        
    plt.plot(epochs_range, train_loss_history, label='Training Loss')
    plt.plot(epochs_range, test_loss_history, label='Test Loss')
        
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Test Loss Curves")
    plt.legend()
    plt.grid(True)
        
    output_filename = "loss_curve.png"
    plt.savefig(output_filename)
    print(f"\nLearning curve saved to {output_filename}")

        # 結果
    print("\n訓練成果摘要:")
        
    y_train_pred = predict(X_train, trained_parameters, layer_dims)
    train_rmse = np.sqrt(np.mean((y_train_pred - y_train)**2))
    print(f"Final Training RMS Error: {train_rmse:.4f}")

    y_test_pred = predict(X_test, trained_parameters, layer_dims)
    test_rmse = np.sqrt(np.mean((y_test_pred - y_test)**2))
    print(f"Final Test RMS Error: {test_rmse:.4f}")