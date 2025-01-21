import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from main import labels, image_data

# parameters for model
IMG_SIZE = (128, 128)
NUM_CLASSES = len(set(labels))
BATCH_SIZE = 32
EPOCHS = 45
LEARNING_RATE = 0.001

# here- normalize image data to be between [0,1]
image_data = image_data / 255.0
print("Shape of image_data:", image_data.shape)  # Debugging: Check the shape of image_data

# Reshape the flattened images back to (num_samples, height, width, channels)
NUM_SAMPLES = image_data.shape[0]  # Number of samples
IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS = 128, 128, 3  # Desired dimensions

if image_data.shape[1] == IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS:
    image_data = image_data.reshape((NUM_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
else:
    raise ValueError("Unexpected data shape. Please check the dimensions of the images.")

# here we convert data to PyTorch tensors and permute to (num_samples, channels, height, width)
X_tensor = torch.tensor(image_data, dtype=torch.float32).permute(0, 3, 1, 2)

y_tensor = torch.tensor(labels, dtype=torch.long)

# standard splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# data is corrupted to prevent overfitting
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class CNN_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (IMG_SIZE[0] // 8) * (IMG_SIZE[1] // 8), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * (IMG_SIZE[0] // 8) * (IMG_SIZE[1] // 8))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# initializing the model, the loss function, and the optimizer
model = CNN_Model(num_classes=NUM_CLASSES)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy for training
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")

# Evaluation on test data
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
print(classification_report(all_labels, all_preds))
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
