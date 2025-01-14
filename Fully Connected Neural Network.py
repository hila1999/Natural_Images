import torch

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from main import labels, image_data

# Parameters
IMG_SIZE = (128, 128)  # Image size
NUM_CLASSES = len(set(labels))  # Number of classes
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Normalize image data (divide by 255 to scale pixel values to [0, 1])
image_data = image_data / 255.0

# Convert data to PyTorch tensors
X_tensor = torch.tensor(image_data, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class FCNN(nn.Module):
    def _init_(self, input_size, num_classes):
        super(FCNN, self)._init_()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),  # First fully connected layer
            nn.ReLU(),  # Activation function
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, 256),  # Second fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Third fully connected layer
            nn.ReLU(),
            nn.Linear(128, 64),  # Fourth fully connected layer
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output layer (raw logits, no softmax)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss function, and optimizer
input_size = IMG_SIZE[0] * IMG_SIZE[1] * 3  # Flattened image size
model = FCNN(input_size=input_size, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("Training the model...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        X_batch, y_batch = batch
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")