import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 150
MODEL_SAVE_PATH = 'cifar_cnn_6layer_gap.pth'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

class_names = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transform_train,
    download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transform_test,
    download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Using 6-Layer CNN with Global Average Pooling (GAP)...")

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.relu6 = nn.ReLU()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_out = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6((self.bn6(self.conv6(x))))
        
        x = self.global_avg_pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc_out(x)
        return x

model = CifarCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()

print("Using SGD Optimizer with Momentum and Weight Decay...")
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

print("Using MultiStepLR Scheduler...")
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[50, 80, 110],
    gamma=0.1
)

train_losses = []
train_accuracies = []
test_accuracies = []
iteration_list = []
count = 0

print(f"Using device: {DEVICE}")
print(f"Start Training on CIFAR-10 (6-Layer CNN + GAP + SGD, {EPOCHS} Epochs)...")

start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    correct_train_epoch = 0
    total_train_epoch = 0
    running_train_loss = 0.0
    
    epoch_start_time = time.time()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        count += 1
        running_train_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train_epoch += labels.size(0)
        correct_train_epoch += (predicted == labels).sum().item()

    avg_train_loss = running_train_loss / len(train_loader)
    epoch_train_accuracy = 100 * correct_train_epoch / total_train_epoch
    
    model.eval()
    test_accuracy = 0.0
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
    
    train_losses.append(avg_train_loss)
    train_accuracies.append(epoch_train_accuracy)
    test_accuracies.append(test_accuracy)
    iteration_list.append(count)
    
    epoch_duration = time.time() - epoch_start_time
    
    print("-" * 50)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Summary (Duration: {epoch_duration:.2f}s):")
    print(f"  Avg Train Loss: {avg_train_loss:.4f}")
    print(f"  Train Accuracy: {epoch_train_accuracy:.2f}%")
    print(f"  Test Accuracy:  {test_accuracy:.2f}%")
    print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 50)

    scheduler.step()

training_duration = time.time() - start_time
print(f"Training Finished! Total time: {training_duration / 60:.2f} minutes")

print(f"Saving model to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved.")

print("Plotting results (Task 1-1)...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(iteration_list, train_losses, label='Training Loss')
plt.title('Learning Curve (6-Layer + GAP + SGD)')
plt.xlabel('Iteration')
plt.ylabel('Loss (Avg per Epoch)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(iteration_list, train_accuracies, label='Training Accuracy', color='blue')
plt.plot(iteration_list, test_accuracies, label='Test Accuracy', color='orange')
plt.title('Accuracy Curve (6-Layer + GAP + SGD)')
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results_curve_cifar_6layer_gap.png')
print("Results curve plot saved to results_curve_cifar_6layer_gap.png")
plt.show()

print("Plotting histograms (Task 1-1)...")
plt.figure(figsize=(15, 12))

def plot_histogram(layer_name, layer_weights, layer_biases, subplot_idx):
    plt.subplot(5, 2, subplot_idx * 2 - 1)
    plt.hist(layer_weights.flatten(), bins=100)
    plt.title(f'Histogram of {layer_name} Weights')
    plt.xlabel('Value')
    plt.ylabel('Number')
    
    plt.subplot(5, 2, subplot_idx * 2)
    plt.hist(layer_biases.flatten(), bins=100)
    plt.title(f'Histogram of {layer_name} Biases')
    plt.xlabel('Value')
    plt.ylabel('Number')

model.to("cpu")
plot_histogram('conv1', model.conv1.weight.data.numpy(), model.conv1.bias.data.numpy(), 1)
plot_histogram('conv3', model.conv3.weight.data.numpy(), model.conv3.bias.data.numpy(), 2)
plot_histogram('conv5', model.conv5.weight.data.numpy(), model.conv5.bias.data.numpy(), 3)
plot_histogram('conv6', model.conv6.weight.data.numpy(), model.conv6.bias.data.numpy(), 4)
plot_histogram('fc_out (output)', model.fc_out.weight.data.numpy(), model.fc_out.bias.data.numpy(), 5)

plt.tight_layout()
plt.savefig('histograms_cifar_6layer_gap.png')
print("Histograms plot saved to histograms_cifar_6layer_gap.png")
plt.show()

model.to(DEVICE)

feature_maps = {
    'conv1': None,
    'conv3': None,
    'conv6': None 
}
def get_activation(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
model.conv3.register_forward_hook(get_activation('conv3'))
model.conv6.register_forward_hook(get_activation('conv6'))
print("Hooks registered on conv1, conv3, and conv6.")

print("Plotting feature maps (Task 1-3)...")

test_loader_no_shuffle = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
images_for_hooks, labels_for_hooks = next(iter(test_loader_no_shuffle))
image_to_test = images_for_hooks[0:1].to(DEVICE)
label_to_test = labels_for_hooks[0].item()

def imshow_cifar(img_tensor):
    img_tensor = img_tensor.cpu().squeeze()
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)

with torch.no_grad():
    output = model(image_to_test)
    _, pred = torch.max(output, 1)

conv1_maps = feature_maps['conv1'].cpu()
conv3_maps = feature_maps['conv3'].cpu()
conv6_maps = feature_maps['conv6'].cpu()

NUM_MAPS_TO_SHOW = 8
fig_hooks, axes_hooks = plt.subplots(4, NUM_MAPS_TO_SHOW, figsize=(NUM_MAPS_TO_SHOW * 2, 9))
fig_hooks.suptitle(f"Feature Maps (Original: {class_names[label_to_test]}, Pred: {class_names[pred.item()]})", fontsize=16)

orig_ax = plt.subplot(4, 1, 1)
imshow_cifar(image_to_test)
orig_ax.set_title(f"Original Image (Label: {class_names[label_to_test]})")
orig_ax.axis('off')

axes_hooks[1, 0].set_ylabel('Conv1 Maps (Shallow)', fontsize=10, rotation=90, labelpad=15)
for i in range(NUM_MAPS_TO_SHOW):
    ax = axes_hooks[1, i]
    if i < conv1_maps.shape[1]:
        ax.imshow(conv1_maps[0, i, :, :], cmap='gray')
    ax.axis('off')

axes_hooks[2, 0].set_ylabel('Conv3 Maps (Mid)', fontsize=10, rotation=90, labelpad=15)
for i in range(NUM_MAPS_TO_SHOW):
    ax = axes_hooks[2, i]
    if i < conv3_maps.shape[1]:
        ax.imshow(conv3_maps[0, i, :, :], cmap='gray')
    ax.axis('off')

axes_hooks[3, 0].set_ylabel('Conv6 Maps (Deep)', fontsize=10, rotation=90, labelpad=15)
for i in range(NUM_MAPS_TO_SHOW):
    ax = axes_hooks[3, i]
    if i < conv6_maps.shape[1]:
        ax.imshow(conv6_maps[0, i, :, :], cmap='gray')
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('feature_maps_cifar_6layer_gap.png')
print("Feature map plot saved to feature_maps_cifar_6layer_gap.png")
plt.show()

print("Plotting classification examples (Task 1-2)...")
correct_examples = []
misclassified_examples = []
NUM_EXAMPLES_TO_SHOW = 5

model.to(DEVICE).eval()
with torch.no_grad():
    for images, labels in test_loader:
        if len(correct_examples) >= NUM_EXAMPLES_TO_SHOW and \
           len(misclassified_examples) >= NUM_EXAMPLES_TO_SHOW:
            break
            
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        for i in range(images.size(0)):
            if len(correct_examples) >= NUM_EXAMPLES_TO_SHOW and \
               len(misclassified_examples) >= NUM_EXAMPLES_TO_SHOW:
                break
            
            image = images[i].cpu()
            label = labels[i].item()
            pred = predicted[i].item()
            
            if pred != label and len(misclassified_examples) < NUM_EXAMPLES_TO_SHOW:
                misclassified_examples.append((image, label, pred))
            elif pred == label and len(correct_examples) < NUM_EXAMPLES_TO_SHOW:
                correct_examples.append((image, label, pred))
                
fig, axes = plt.subplots(2, NUM_EXAMPLES_TO_SHOW, figsize=(15, 7))
fig.suptitle('Model Classification Examples (CIFAR-10, 6-Layer + GAP)', fontsize=16)

for i in range(NUM_EXAMPLES_TO_SHOW):
    if i < len(correct_examples):
        image, label, pred = correct_examples[i]
        ax = axes[0, i]
        imshow_cifar(image)
        ax.set_title(f"Label: {class_names[label]}\nPred: {class_names[pred]}", color='green')
        ax.axis('off')
axes[0, 0].set_ylabel('Correctly Classified', fontsize=12, rotation=90, labelpad=25)

for i in range(NUM_EXAMPLES_TO_SHOW):
    if i < len(misclassified_examples):
        image, label, pred = misclassified_examples[i]
        ax = axes[1, i]
        imshow_cifar(image)
        ax.set_title(f"Label: {class_names[label]}\nPred: {class_names[pred]}", color='red')
        ax.axis('off')
axes[1, 0].set_ylabel('Misclassified', fontsize=12, rotation=90, labelpad=25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('classification_results_cifar_6layer_gap.png')
print("Classification results plot saved to classification_results_cifar_6layer_gap.png")
plt.show()

print("All tasks for 6-Layer CNN with GAP complete.")