# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:23:29 2023

@author: he_98
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image


class TrafficSignsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.root_dir + '/' + self.data['Path'][idx]
        image = Image.open(image_path).convert('RGB')
        label = self.data['class'][idx]
        if self.transform:
            image = self.transform(image)
            return image, label


# Set the path to your CSV file and image folder
csv_file_train = 'Traffic_Sign_Dataset/Traffic_Sign_Dataset/Train.csv'
csv_file_test = 'Traffic_Sign_Dataset/Traffic_Sign_Dataset/Test.csv'

image_folder = 'Traffic_Sign_Dataset/Traffic_Sign_Dataset'

# Define the transformation to apply to the training data
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the images to a consistent size
    transforms.ToTensor(),  # Converts the images to PyTorch tensors
    # Add any additional transformations here
])

# Create the dataset
train_dataset = TrafficSignsDataset(csv_file=csv_file_train, root_dir=image_folder, transform=transform)

print(train_dataset)
test_dataset = TrafficSignsDataset(csv_file=csv_file_test, root_dir=image_folder, transform=transform)
print(test_dataset)


# Create a data loader to efficiently load the data during training
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

num_classes = 43

# Define your CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # Adjusted dimensions
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        
        self.dropout3 = nn.Dropout(0.5)  # Additional dropout layer

        
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool1(out)
        out = self.dropout1(out)

        
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.maxpool2(out)
        out = self.dropout2(out)  # Apply the additional dropout layer
        
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu5(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        
        return out

# Create an instance of your CNN model
#num_classes = 43  # Replace with the number of classes in your dataset
model = CNNModel(num_classes)

# Define your loss function
criterion = nn.CrossEntropyLoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train your model
num_epochs = 10

for epoch in range(num_epochs):
    # Training loop
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    validation_accuracy = correct / total
    # Print training/validation metrics for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Accuracy: {validation_accuracy}")

torch.save(model.state_dict(), 'model1.pth')
