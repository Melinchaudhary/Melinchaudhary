import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the MNIST dataset (or your custom dataset of letters)
mnist = cv2.imread('path_to_mnist_dataset/mnist.png', 0)
labels = np.arange(10)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mnist, labels, test_size=0.2, random_state=42)

# Reshape the data to 1D arrays
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Create and train the Multi-Layer Perceptron (MLP) classifier
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

