import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
#和Q1-1-1除了weight decay都一樣
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 20 

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform, 
    download=True
)


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU() 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1) 
        
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) 
        return x

model = myCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 

train_losses = []
train_accuracies = []
test_accuracies = []
iteration_list = []
count = 0


for epoch in range(EPOCHS):
    model.train() 
    correct_train = 0
    total_train = 0
    
    for i, (images, labels) in enumerate(train_loader):

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()
        count += 1
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0:

            train_loss = loss.item()
            train_accuracy = 100 * correct_train / total_train

            model.eval() 
            with torch.no_grad(): 
                correct_test = 0
                total_test = 0
                for test_images, test_labels in test_loader:
                    test_images, test_labels = test_images.to(DEVICE), test_labels.to(DEVICE)
                    test_outputs = model(test_images)
                    _, test_predicted = torch.max(test_outputs.data, 1)
                    total_test += test_labels.size(0)
                    correct_test += (test_predicted == test_labels).sum().item()
            
            test_accuracy = 100 * correct_test / total_test

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            iteration_list.append(count)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Iteration [{count}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"  Test Accuracy: {test_accuracy:.2f}%")
            
            model.train()
            correct_train = 0
            total_train = 0

print("Finished!")

#
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(iteration_list, train_losses, label='Training Loss')
plt.title('Learning Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(iteration_list, train_accuracies, label='Training Accuracy', color='blue')
plt.plot(iteration_list, test_accuracies, label='Test Accuracy', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
results_plot_path = 'results_curve-L2.png'
plt.savefig(results_plot_path)
plt.show()
MODEL_SAVE_PATH = 'cnnL2.pth'
torch.save(model.state_dict(), MODEL_SAVE_PATH)

plt.figure(figsize=(15, 10))

def plot_histogram(layer_name, layer_weights, layer_biases, subplot_idx):
    plt.subplot(4, 2, subplot_idx * 2 - 1)
    plt.hist(layer_weights.flatten(), bins=100)
    plt.title(f'Histogram of {layer_name} Weights')
    plt.xlabel('Value')
    plt.ylabel('Number')
    
    plt.subplot(4, 2, subplot_idx * 2)
    plt.hist(layer_biases.flatten(), bins=100)
    plt.title(f'Histogram of {layer_name} Biases')
    plt.xlabel('Value')
    plt.ylabel('Number')

model.to("cpu")
conv1_weights = model.conv1.weight.data.numpy()
conv1_biases = model.conv1.bias.data.numpy()
conv2_weights = model.conv2.weight.data.numpy()
conv2_biases = model.conv2.bias.data.numpy()
fc1_weights = model.fc1.weight.data.numpy()
fc1_biases = model.fc1.bias.data.numpy()
fc2_weights = model.fc2.weight.data.numpy()
fc2_biases = model.fc2.bias.data.numpy()

plot_histogram('conv1', conv1_weights, conv1_biases, 1)
plot_histogram('conv2', conv2_weights, conv2_biases, 2)
plot_histogram('fc1 (dense1)', fc1_weights, fc1_biases, 3)
plot_histogram('fc2 (output)', fc2_weights, fc2_biases, 4)
plt.tight_layout()
histogram_plot_path = 'histograms-L2.png'
plt.savefig(histogram_plot_path)
plt.show()
