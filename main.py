# from torchvision import datasets, transforms
#
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
#
#
#
#
# # Define transformations (resize, normalize, etc.)
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize to 128x128
#     transforms.ToTensor()          # Convert to tensor
# ])
#
# # Load the dataset
# dataset = datasets.ImageFolder(root="natural_images", transform=transform)
#
# # Print the mapping between class indices and folder names
# print("Class-to-Index Mapping:", dataset.class_to_idx)
#
#
# # Function to display an image with its label
# def show_image(image, label, class_to_idx):
#     plt.imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
#     plt.title(f"Label: {label} (Class: {list(class_to_idx.keys())[label]})")
#     plt.axis('off')
#     plt.show()
#
# # Display first 5 images
# for i in range(5):
#     image, label = dataset[i]
#     show_image(image, label, dataset.class_to_idx)
# # Iterate over the dataset and print the image path and its label
# for i in range(5):  # Print first 5 images for verification
#     image, label = dataset[i]
#     print(f"Image {i}: Label {label} (Class: {list(dataset.class_to_idx.keys())[label]})")
# # Example of accessing the first image and its label
# image, label = dataset[0]
# print(f"Image size: {image.size()}, Label: {label}")
#
#


import os

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path to the dataset folder
DATASET_PATH = "natural_images"

# Define image size for resizing
IMG_SIZE = (128, 128)  # Resize images to 64x64 for simplicity

# Initialize lists for images and labels
image_data = []
labels = []

# Load dataset
print("Loading dataset...")
for label, category in enumerate(os.listdir(DATASET_PATH)):
    category_path = os.path.join(DATASET_PATH, category)
    if os.path.isdir(category_path):
        print(f"Processing category: {category}")
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            try:
                # Open and resize the image
                img = Image.open(image_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                image_data.append(np.array(img).flatten())  # Flatten the image
                labels.append(label)  # Assign numeric label for the category
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

# Convert to NumPy arrays
image_data = np.array(image_data)
labels = np.array(labels)

print(f"Loaded {len(image_data)} images with {len(set(labels))} classes.")



# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)


# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define a dummy classifier (e.g., most frequent class)
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = dummy_clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test,y_pred))


def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Logistic Regression model.

    Parameters:
    X_train, X_test: Training and testing features.
    y_train, y_test: Training and testing labels.
    """
    # Initialize the Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)

    # Train the model
    print("\nTraining Logistic Regression model...")
    log_reg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = log_reg.predict(X_test)

    # Evaluate the model
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# Call the function to train and evaluate the model
train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test)