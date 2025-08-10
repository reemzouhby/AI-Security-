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

model = tf.keras.models.load_model("mnist_model.h5")

# Create ART KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))



# Generate adversarial examples
attack = FastGradientMethod(estimator=classifier, eps=0.3)  # eps  small

x_test_adv = attack.generate(x=test_images)

# Save adversarial examples and their labels
np.save("x_test_adv.npy", x_test_adv)
np.save("test_labels.npy", test_labels)

print("FGSM adversarial examples generated and saved.")


