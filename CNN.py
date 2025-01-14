import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from main import labels, image_data

# Parameters
IMG_SIZE = (224, 224)  # VGG16 expects 224x224 images
NUM_CLASSES = len(set(labels))
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

# Normalize image data
image_data = image_data / 255.0

# Resize images to 224x224
image_data_resized = torch.nn.functional.interpolate(
    torch.tensor(image_data).permute(0, 3, 1, 2), size=IMG_SIZE, mode='bilinear'
).permute(0, 2, 3, 1).numpy()

# Convert data to PyTorch tensors
X_tensor = torch.tensor(image_data_resized, dtype=torch.float32)
# Check the shape of X_tensor
print("Original shape of X_tensor:", X_tensor.shape)

# Reshape to (batch_size, channels, height, width)
if len(X_tensor.shape) == 4 and X_tensor.shape[3] == 3:  # (num_samples, height, width, channels)
    X_tensor = X_tensor.permute(0, 3, 1, 2)  # Convert to (num_samples, channels, height, width)
else:
    raise ValueError("Expected input shape: (num_samples, height, width, channels)")

print("Reshaped X_tensor:", X_tensor.shape)
y_tensor = torch.tensor(labels, dtype=torch.long)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Normalization for testing
test_transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformations
X_train = torch.stack([train_transform(img) for img in X_train])
X_test = torch.stack([test_transform(img) for img in X_test])

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Freeze all layers
for param in vgg16.parameters():
    param.requires_grad = False

# Replace the classifier part to match the number of classes
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, NUM_CLASSES)

# Unfreeze the classifier layers
for param in vgg16.classifier.parameters():
    param.requires_grad = True

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    vgg16.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = vgg16(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Evaluation
vgg16.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = vgg16(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(f"Hila is loser")
