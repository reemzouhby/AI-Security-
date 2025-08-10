from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # by default 85% training and  15% for testing 60000 for training and 10000 for teting

train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']

trainer = tf.keras.models.load_model('mnist_trained_model.h5')

# Create ART KerasClassifier
classifierr = KerasClassifier(model=trainer, clip_values=(0, 1))

attack = FastGradientMethod(estimator=classifierr, eps=0.4)  # eps  small

x_test_adv = attack.generate(x=test_images)

# Evaluate the model on clean and adversarial examples
test_images_reshaped = x_test_adv.reshape((-1, 28, 28, 1))
test_images=test_images.reshape((-1,28,28,1 ))
loss_clean, accuracy_clean = trainer.evaluate(test_images, test_labels, verbose=0)
loss_adv, accuracy_adv =trainer.evaluate(test_images_reshaped, test_labels, verbose=0)

print(f"Accuracy on clean test examples: {accuracy_clean:.4f}")
print(f"Accuracy on adversarial test examples: {accuracy_adv:.4f}")
acc_Diff = accuracy_clean-accuracy_adv
print(f"Accuracy drop due to adversarial attack: {acc_Diff:.4f}")
# Add a channel dimension for grayscale images


# Get model predictions on test images
predictions = trainer.predict(test_images_reshaped)

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

