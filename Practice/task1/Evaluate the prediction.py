from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
model = tf.keras.models.load_model("mnist_model.h5")
# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
test_images=test_images.reshape(test_images.shape[0], 28, 28, 1)
x_test_adv = np.load("x_test_adv.npy")
y_test = np.load("test_labels.npy")
class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
# Evaluate the model on clean and adversarial examples
test_images_reshaped = x_test_adv.reshape((test_images.shape[0], 28, 28, 1))
loss_clean, accuracy_clean = model.evaluate(test_images, y_test, verbose=0)
loss_adv, accuracy_adv = model.evaluate(test_images_reshaped, y_test, verbose=0)

print(f"Accuracy on clean test examples: {accuracy_clean:.4f}")
print(f"Accuracy on adversarial test examples: {accuracy_adv:.4f}")
acc_Diff = accuracy_clean-accuracy_adv
print(f"Accuracy drop due to adversarial attack: {acc_Diff:.4f}")
# Add a channel dimension for grayscale images


# Get model predictions on test images
predictions = model.predict(test_images_reshaped)

# Each prediction is a vector of 10 scores → pick the class with the highest score
predicted_labels = np.argmax(predictions, axis=1)
plt.figure(figsize=(10, 5))
for i in range(50):
    plt.subplot(5, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_reshaped[i], cmap=plt.cm.gray)
    true_label = class_names[test_labels[i]]
    predicted_label = class_names[predicted_labels[i]]
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
plt.tight_layout()
plt.show()

print(f"Number of correct predictions: {np.sum(np.argmax(predictions, axis=1) == test_labels)}")
print(f"Number of  wrong prediction : {len(test_labels) - np.sum(np.argmax(predictions, axis=1) == test_labels)}")
print(f"Total test samples: {len(test_labels)}")

