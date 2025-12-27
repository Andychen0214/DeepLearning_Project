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

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A, Z

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

def compute_categorical_cross_entropy_loss(AL, Y):
    m = Y.shape[1]
    loss = - (1/m) * np.sum(Y * np.log(AL + 1e-8))
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

def backward_propagation(AL, Y, caches, layer_activations_backward):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dZ_last = AL - Y 
    linear_cache_last, _ = caches[L-1]
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_module_backward(dZ_last, linear_cache_last)

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


def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)].T

def load_and_preprocess_softmax_data(file_path, target_column_index, test_split_ratio=0.2):
    try:
        data = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return None, None, None, None, None, None
    
    data.drop(columns=[1], inplace=True)
    
    data[target_column_index] = data[target_column_index].apply(lambda x: 1 if x == 'g' else 0)
    
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = data.drop(columns=[target_column_index])
    y_labels = data[[target_column_index]]

    split_index = int(len(data) * (1 - test_split_ratio))
    X_train_raw, X_test_raw = X.iloc[:split_index], X.iloc[split_index:]
    y_train_labels, y_test_labels = y_labels.iloc[:split_index], y_labels.iloc[split_index:]
    
    train_mean = X_train_raw.mean()
    train_std = X_train_raw.std()
    X_train = (X_train_raw - train_mean) / train_std
    X_test = (X_test_raw - train_mean) / train_std
    
    X_train_np = X_train.to_numpy()
    y_train_labels_np = y_train_labels.to_numpy()
    X_test_np = X_test.to_numpy()
    y_test_labels_np = y_test_labels.to_numpy()

    y_train_one_hot = to_one_hot(y_train_labels_np, 2)
    y_test_one_hot = to_one_hot(y_test_labels_np, 2)

    return X_train_np, y_train_one_hot, X_test_np, y_test_one_hot, y_train_labels_np, y_test_labels_np

def train_softmax_model(X_train, y_train_one_hot, X_test, y_test_one_hot, layer_dims, epochs, learning_rate):
    parameters = initialize_parameters(layer_dims)
    layer_activations = [relu] * (len(layer_dims) - 2) + [softmax]
    layer_activations_backward = [relu_backward] * (len(layer_dims) - 2)
    
    train_loss_history = []
    test_loss_history = []

    print(f"Starting training for {epochs} epochs with learning rate {learning_rate}...")
    print(f"Network Architecture: {layer_dims}")

    for i in range(epochs):
        # 訓練
        AL, caches = forward_propagation(X_train.T, parameters, layer_activations)
        grads = backward_propagation(AL, y_train_one_hot, caches, layer_activations_backward)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # 每隔一段時間記錄並打印 train 和 test loss
        if i % 10000 == 0 or i == epochs - 1:
            # 計算 Training Loss
            train_loss = compute_categorical_cross_entropy_loss(AL, y_train_one_hot)
            train_loss_history.append(train_loss)
            
            AL_test, _ = forward_propagation(X_test.T, parameters, layer_activations)
            test_loss = compute_categorical_cross_entropy_loss(AL_test, y_test_one_hot)
            test_loss_history.append(test_loss)
            
            print(f"Epoch {i}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            
    return parameters, train_loss_history, test_loss_history

def predict_softmax(X, parameters, layer_dims):
    layer_activations = [relu] * (len(layer_dims) - 2) + [softmax]
    probs, _ = forward_propagation(X.T, parameters, layer_activations)
    predictions = np.argmax(probs, axis=0).reshape(1, -1)
    return predictions.T

def compute_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

def plot_confusion_matrix_matplotlib(ax, cm, class_labels, title):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_labels, yticklabels=class_labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")


if __name__ == "__main__":
    
    DATASET_PATH = "2025_ionosphere_data.csv"
    TARGET_COLUMN_INDEX = 34
    
    X_train, y_train_one_hot, X_test, y_test_one_hot, y_train_labels, y_test_labels = load_and_preprocess_softmax_data(
        file_path=DATASET_PATH,
        target_column_index=TARGET_COLUMN_INDEX,
        test_split_ratio=0.2
    )

    if X_train is not None:
        
        # 使用 [33, 17, 2] 架構
        layer_dims = [X_train.shape[1], 17, 2] 
        EPOCHS = 200000
        LEARNING_RATE = 0.0005
        
        trained_parameters, train_loss_history, test_loss_history = train_softmax_model(
            X_train, y_train_one_hot, X_test, y_test_one_hot,
            layer_dims, EPOCHS, LEARNING_RATE
        )

        plt.figure(figsize=(10, 6))
        log_interval = 10000
        epochs_range = np.arange(len(train_loss_history)) * log_interval
        plt.plot(epochs_range, train_loss_history, label='Training Loss')
        plt.plot(epochs_range, test_loss_history, label='Test Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss (Categorical Cross-Entropy)")
        plt.title("Learning Curve for Softmax Classification")
        plt.legend()
        plt.grid(True)
        plt.savefig("softmax_loss_curve.png")
        print(f"\nLearning curve saved to softmax_loss_curve.png")

        
        y_train_pred = predict_softmax(X_train, trained_parameters, layer_dims)
        train_accuracy = np.mean(y_train_pred == y_train_labels) * 100
        print(f"Final Training Accuracy: {train_accuracy:.2f}%")

        y_test_pred = predict_softmax(X_test, trained_parameters, layer_dims)
        test_accuracy = np.mean(y_test_pred == y_test_labels) * 100
        print(f"Final Test Accuracy: {test_accuracy:.2f}%")
        
        cm_train = compute_confusion_matrix(y_train_labels, y_train_pred)
        cm_test = compute_confusion_matrix(y_test_labels, y_test_pred)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        class_labels = ['Bad (0)', 'Good (1)']
        
        plot_confusion_matrix_matplotlib(ax[0], cm_train, class_labels, 'Training Set: Confusion Matrix')
        plot_confusion_matrix_matplotlib(ax[1], cm_test, class_labels, 'Test Set: Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig("softmax_confusion_matrix_matplotlib.png")
        print("Confusion matrix plots saved to softmax_confusion_matrix_matplotlib.png")