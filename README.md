# Deep Learning Practice 🚀

This repository serves as a record of my deep learning implementation practice during my graduate studies.
The project spans from building neural networks from scratch using **NumPy** to implementing various classic models using **PyTorch**.

## 👤 Author
- **GUAN-RU CHEN**
- Institute of Electrical and Control Engineering, National Yang Ming Chiao Tung University (NYCU)

## Tech Stack
- **Language**: Python 3.x
- **Framework**: PyTorch
- **Libraries**: NumPy, Pandas, Matplotlib, Scikit-Learn

## ⚠️ Note on Datasets
**This project does not contain any datasets.**
- Standard datasets (like MNIST, CIFAR-10) will be automatically downloaded by PyTorch (`download=True`) when you run the code for the first time.
- For custom datasets (e.g., Shakespeare text files, CSV data), you need to download and place them in the correct directory manually.

## Contents

This repository contains three main modules covering fundamental deep learning models:

### 1. Regression & Classification
Building neural networks from scratch (NumPy Only) to understand the principles of Forward and Backward Propagation.
- **Regression**: Predicting energy efficiency (Heating Load Dataset).
- **Classification**: Classification of radar returns from the ionosphere (Ionosphere Dataset).
- **Core Concepts**: Forward/Backward Propagation, Gradient Descent, Activation Functions (ReLU, Softmax).

### 2. Convolutional Neural Networks (CNN)
Implementing CNNs using PyTorch for image recognition tasks.
- **MNIST**: Handwritten Digit Recognition (Basic CNN).
- **CIFAR-10**: Object Recognition (Deep CNN with 6 layers).
- **Advanced Techniques**:
  - L2 Regularization (Weight Decay)
  - Global Average Pooling (GAP)
  - Feature Map Visualization (Hook mechanism)
  - Confusion Matrix Analysis

### 3. Sequence Models (RNN & LSTM)
Handling time-series data and text generation tasks.
- **Char-RNN**: Basic character-level text generation.
- **LSTM**: Long Short-Term Memory networks, addressing the vanishing gradient problem in long sequences.
- **Task**: Automated text generation mimicking Shakespeare's style.
- **Experiment**: Sensitivity analysis comparing the impact of **Hidden Size**, **Embedding Size**, and **Sequence Length** on model performance.

## Results
The project includes visualization of **Loss/Accuracy learning curves** and **Weight Distribution Histograms** to analyze potential overfitting or gradient anomalies during training to create the best model.

---
*Created for self-learning and research purposes.*