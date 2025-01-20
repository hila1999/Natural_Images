# fcnn - version with early stopping implementation
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from main import labels, image_data

IMG_SIZE = (128, 128)  # Image size after we resized
NUM_CLASSES = len(set(labels))
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 0.001
PATIENCE = 5  # patience for Early stopping, num of epochs to wait before stopping

# normalizing image data (divide by 255 to scale pixel values to [0, 1])
image_data = image_data / 255.0

X_tensor = torch.tensor(image_data, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)

# splitting dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# fully connected neural network implementation
class FCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# model, loss function, and optimizer initialization
input_size = IMG_SIZE[0] * IMG_SIZE[1] * 3  # Flattened image size
model = FCNN(input_size=input_size, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# variables for Early stopping
best_val_loss = float('inf')
patience_counter = 0

# training loop below
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
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

    # validation step
    model.eval()
    val_loss = 0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            X_val_batch, y_val_batch = batch
            val_outputs = model(X_val_batch)
            val_loss += criterion(val_outputs, y_val_batch).item()
            val_preds = torch.argmax(val_outputs, dim=1)
            all_val_preds.extend(val_preds.cpu().numpy())
            all_val_labels.extend(y_val_batch.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy_score(all_val_labels, all_val_preds):.4f}")

    # early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break  # stops training if patience is exceeded

# final evaluation, on the test set, then the printing
model.eval()
all_test_preds = []
all_test_labels = []
with torch.no_grad():
    for batch in test_loader:
        X_test_batch, y_test_batch = batch
        test_outputs = model(X_test_batch)
        test_preds = torch.argmax(test_outputs, dim=1)
        all_test_preds.extend(test_preds.cpu().numpy())
        all_test_labels.extend(y_test_batch.cpu().numpy())

print("\nTest Classification Report:")
print(classification_report(all_test_labels, all_test_preds, zero_division=0))
print(f"Test Accuracy: {accuracy_score(all_test_labels, all_test_preds):.4f}")
