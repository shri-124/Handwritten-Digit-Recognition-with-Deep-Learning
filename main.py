

import gzip
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns



DATASET_DIR = "dataset"
MNIST_TRAIN_IMS_GZ = os.path.join(DATASET_DIR, "train-images-idx3-ubyte.gz")
MNIST_TRAIN_LBS_GZ = os.path.join(DATASET_DIR, "train-labels-idx1-ubyte.gz")
MNIST_TEST_IMS_GZ = os.path.join(DATASET_DIR, "t10k-images-idx3-ubyte.gz")
MNIST_TEST_LBS_GZ = os.path.join(DATASET_DIR, "t10k-labels-idx1-ubyte.gz")

NROWS = 28
NCOLS = 28


def load_data():
    print("Unpacking training images ...")
    with gzip.open(MNIST_TRAIN_IMS_GZ, mode='rb') as f:
        magic_num, train_sz, nrows, ncols = struct.unpack('>llll', f.read(16))
        print("magic number: %d, num of examples: %d, rows: %d, columns: %d" % (magic_num, train_sz, nrows, ncols))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * train_sz * nrows * ncols, data_bn)
        train_ims = np.asarray(data)
        train_ims = train_ims.reshape(train_sz, nrows * ncols)
    print("~" * 5)

    print("Unpacking training labels ...")
    with gzip.open(MNIST_TRAIN_LBS_GZ, mode='rb') as f:
        magic_num, train_sz = struct.unpack('>ll', f.read(8))
        print("magic number: %d, num of examples: %d" % (magic_num, train_sz))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * train_sz, data_bn)
        train_lbs = np.asarray(data)
    print("~" * 5)

    print("Unpacking test images ...")
    with gzip.open(MNIST_TEST_IMS_GZ, mode='rb') as f:
        magic_num, test_sz, nrows, ncols = struct.unpack('>llll', f.read(16))
        print("magic number: %d, num of examples: %d, rows: %d, columns: %d" % (magic_num, test_sz, nrows, ncols))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * test_sz * nrows * ncols, data_bn)
        test_ims = np.asarray(data)
        test_ims = test_ims.reshape(test_sz, nrows * ncols)
    print("~" * 5)

    print("Unpacking test labels ...")
    with gzip.open(MNIST_TEST_LBS_GZ, mode='rb') as f:
        magic_num, test_sz = struct.unpack('>ll', f.read(8))
        print("magic number: %d, num of examples: %d" % (magic_num, test_sz))
        data_bn = f.read()
        data = struct.unpack('<' + 'B' * test_sz, data_bn)
        test_lbs = np.asarray(data)
    print("~" * 5)
    return train_ims, train_lbs, test_ims, test_lbs



# Define the 3-layer fully connected neural network
# class FullyConnectedNN(nn.Module):
#     def __init__(self):
#         super(FullyConnectedNN, self).__init__()
#         self.network = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 10)
#         )

#     def forward(self, x):
#         return self.network(x)

# A simple CNN model you can use:
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)
    
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

train_ims, train_lbs, test_ims, test_lbs = load_data()

# Convert numpy arrays to PyTorch tensors
train_images_tensor = torch.tensor(train_ims).float()
train_labels_tensor = torch.tensor(train_lbs).long()
test_images_tensor = torch.tensor(test_ims).float()
test_labels_tensor = torch.tensor(test_lbs).long()

# Reshape to match CNN expected input: [batch_size, 1, 28, 28]
train_images_tensor = train_images_tensor.view(-1, 1, 28, 28)
test_images_tensor = test_images_tensor.view(-1, 1, 28, 28)

# Create DataLoader instances for training and testing
batch_size = 64
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer by calling the class
model = model = CNNModel()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # found this online

# Training function
def train_model(model, train_loader, loss_function, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def show_confusion_matrix(model, test_loader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images)
            _, predicted = torch.max(preds, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Train and evaluate the model
train_model(model, train_loader, loss_function, optimizer)
accuracy = evaluate_model(model, test_loader)
print(f'Test Accuracy: {accuracy}%')

# Show confusion matrix
show_confusion_matrix(model, test_loader)

# Save predictions to SQLite database
import sqlite3

def save_predictions_to_sql(model, test_loader, db_path="mnist_predictions.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            true_label INTEGER,
            predicted_label INTEGER,
            correct BOOLEAN
        )
    ''')

    model.eval()
    with torch.no_grad():
        id_counter = 0
        for images, labels in test_loader:
            preds = model(images)
            _, predicted = torch.max(preds, 1)
            for true, pred in zip(labels.numpy(), predicted.numpy()):
                correct = int(true == pred)
                cursor.execute('INSERT INTO predictions (id, true_label, predicted_label, correct) VALUES (?, ?, ?, ?)',
                               (id_counter, int(true), int(pred), correct))
                id_counter += 1

    conn.commit()
    conn.close()

# Call the function to save the predictions
save_predictions_to_sql(model, test_loader)


# Save the trained model
torch.save(model.state_dict(), 'cnn_mnist_model.pth')
print("Model saved to cnn_mnist_model.pth")