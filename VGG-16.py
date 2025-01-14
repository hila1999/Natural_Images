# Libraries for plotting and visualizations
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim

matplotlib.use('TkAgg')  # Switch to a backend that supports interactive plotting

# General libraries
import os
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch libraries for dataset handling and data augmentation
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms ,models



data_dir = "natural_images"

classes = []
img_classes = []
n_image = []
height = []
width = []
dim = []

# Collecting data information
for folder in os.listdir(data_dir):
    classes.append(folder)

    images = os.listdir(os.path.join(data_dir, folder))
    n_image.append(len(images))

    for img_name in images:
        img_path = os.path.join(data_dir, folder, img_name)
        img = np.array(Image.open(img_path))
        img_classes.append(folder)
        height.append(img.shape[0])
        width.append(img.shape[1])
        dim.append(img.shape[2] if len(img.shape) == 3 else 1)

# Class-level DataFrame
df = pd.DataFrame({
    'Class': classes,
    'Number of Images': n_image
})

# Image-level DataFrame
image_df = pd.DataFrame({
    "classes": img_classes,
    "height": height,
    "width": width
})


print("Random heights: ", height[10], height[123])
print("Random widths: ", width[10], width[123])
print("Classes and number of images:")
print(df)
# Grouped statistical description by classes
img_df = image_df.groupby("classes").describe()
print("Statistical summary of image dimensions per class:")
print(img_df)

# Visualize number of images per class
sns.barplot(x='Class', y='Number of Images', data=df)
plt.title('Number of images per class')
plt.show()

#Display random image
img=Image.open('natural_images/cat/cat_0006.jpg')
plt.imshow(img)
plt.axis('off')  # Remove axes for a cleaner display
plt.title("Class: Cat, Image: cat_0007.jpg")  # Optional title
plt.show()



# Define the transformations
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.95, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet standards
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the full dataset
all_data = datasets.ImageFolder(root=data_dir, transform=image_transforms['train'])

# Split the data
train_data_len = int(len(all_data) * 0.8)
valid_data_len = int((len(all_data) - train_data_len) / 2)
test_data_len = int(len(all_data) - train_data_len - valid_data_len)
train_data, val_data, test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])

# Apply transformations
train_data.dataset.transform = image_transforms['train']
val_data.dataset.transform = image_transforms['val']
test_data.dataset.transform = image_transforms['test']

# Print lengths of splits
print(f"Train data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")
print(f"Test data length: {len(test_data)}")

# Create DataLoader
batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)  # No need to shuffle validation data
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)  # No need to shuffle test data

trainiter = iter(train_loader)
features, labels = next(trainiter)
print(features.shape, labels.shape)

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
print(model)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False


# Define the number of output classes (8 in your case)
n_classes = 8
n_inputs = model.classifier[6].in_features  # This should be 4096 for VGG16

# Modify the classifier part of the model
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256),  # First fully connected layer
    nn.ReLU(),                 # ReLU activation
    nn.Dropout(0.4),           # Dropout layer for regularization
    nn.Linear(256, n_classes), # Output layer with n_classes outputs
    nn.LogSoftmax(dim=1)       # LogSoftmax activation for multi-class classification
)

# Print the modified classifier
print(model.classifier)

# Show the summary of our model (if torchsummary is available)
# This would show the model summary, but 'torchsummary' is not installed in Kaggle environments
# summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')

# Define the loss function and optimizer
criterion = nn.NLLLoss()  # Negative Log-Likelihood loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer



# Show the model's architecture
print(model)

model.class_to_idx = all_data.class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

list(model.idx_to_class.items())

def train(model,
          criterion,
          optimizer,
          train_loader,
          val_loader,
          save_location,
          early_stop=3,
          n_epochs=20,
          print_every=2):
    # Initializing some variables
    valid_loss_min = np.inf
    stop_count = 0
    valid_max_acc = 0
    history = []
    model.epochs = 0

    # Loop starts here
    for epoch in range(n_epochs):
        train_loss = 0
        valid_loss = 0
        train_acc = 0
        valid_acc = 0

        model.train()  # Set the model to training mode
        ii = 0

        # Training loop
        for data, label in train_loader:
            ii += 1
            # For CPU (remove .cuda())
            data, label = data, label
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(label.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item() * data.size(0)

            if ii % 15 == 0:
                print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete.')

        model.epochs += 1

        # Validation loop
        with torch.no_grad():
            model.eval()
            for data, label in val_loader:
                # For CPU (remove .cuda())
                data, label = data, label
                output = model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)

                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(label.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                valid_acc += accuracy.item() * data.size(0)

        # Compute average loss and accuracy for train and validation
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(val_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        valid_acc = valid_acc / len(val_loader.dataset)

        history.append([train_loss, valid_loss, train_acc, valid_acc])

        # Print every `print_every` epochs
        if (epoch + 1) % print_every == 0:
            print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
            print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')

        # Check if validation loss improved and save the model
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_location)
            stop_count = 0
            valid_loss_min = valid_loss
            valid_max_acc = valid_acc
            best_epoch = epoch
        else:
            stop_count += 1

        # Early stopping condition
        if stop_count >= early_stop:
            print(
                f'\nEarly Stopping Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
            model.load_state_dict(torch.load(save_location))  # Load the best model
            history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
            return model, history

    model.optimizer = optimizer
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')

    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


model, history = train(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    save_location='Natural_Images/vgg16_model.pt',
    early_stop=5,
    n_epochs=30,
    print_every=2)

print(history)
