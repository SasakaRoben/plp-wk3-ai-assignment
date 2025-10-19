# Task2_MNIST_CNN.py (or use in Jupyter cells)
"""
Deep Learning Task:
- Load MNIST dataset
- Build CNN model
- Train and evaluate (target: >95% accuracy)
- Visualize sample predictions
"""

# -------------------------
# 1) Imports
# -------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------
# 2) Device configuration
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# 3) Data preparation
# -------------------------
# Transform: convert to tensor and normalize to range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load training and test sets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# -------------------------
# 4) Define CNN architecture
# -------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # input: 1x28x28 -> output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # input: 32x14x14 -> output: 64x14x14
        self.pool = nn.MaxPool2d(2, 2)                # reduces spatial dimension by 2
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected + dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Instantiate model and move to device
model = CNNModel().to(device)
print(model)

# -------------------------
# 5) Define loss and optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# 6) Training loop
# -------------------------
epochs = 5  # typically 5-10 epochs achieve >95% accuracy
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

print("Training complete.")

# -------------------------
# 7) Evaluation
# -------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# -------------------------
# 8) Visualize predictions on 5 sample images
# -------------------------
def show_predictions(model, test_loader, num_samples=5):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Predict
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Plot first few images with predictions
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}")
        plt.axis('off')
    plt.show()

show_predictions(model, test_loader)
