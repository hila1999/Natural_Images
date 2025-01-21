import os
import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# path to dataset
DATASET_PATH = "natural_images"

# the size of images we'll resize to
IMG_SIZE = (128, 128)

image_data = []
labels = []

# load dataset loop
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

# convertion to NumPy arrays
image_data = np.array(image_data)
labels = np.array(labels)

print(f"Loaded {len(image_data)} images with {len(set(labels))} classes.")



# splitting of dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)


# normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# a dummy classifier by most frequent class
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

# evaluate the classifier
y_pred = dummy_clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test,y_pred))


def logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Logistic Regression - softmax - model.

    parameters:
    X_train, X_test: Training and testing features.
    y_train, y_test: Training and testing labels. (the images classes)
    """
    # Initialize the Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)

    # model training
    print("\nTraining Logistic Regression model...")
    log_reg.fit(X_train, y_train)

    # predict on the test set
    y_pred = log_reg.predict(X_test)

    # evaluate the model
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# function call to train and evaluate the model
logistic_regression(X_train, X_test, y_train, y_test)
