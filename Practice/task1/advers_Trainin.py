from art.attacks.evasion import FastGradientMethod
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd

tf.config.run_functions_eagerly(True)

from art.estimators.classification import KerasClassifier
import warnings



warnings.filterwarnings('ignore')
from art.defences.trainer import AdversarialTrainer
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # by default 85% training and  15% for testing 60000 for training and 10000 for teting

train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
model = tf.keras.models.load_model("mnist_model.h5")

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


classifier = KerasClassifier(model=model, clip_values=(0, 1))
attack = FastGradientMethod(estimator=classifier, eps=0.3)
trainer = AdversarialTrainer(classifier, attack, ratio=1.0)
trainer.fit(train_images, train_labels, nb_epochs=10, batch_size=128)
# check the accuracy after added ana dversarial examples
train_loss, train_acc = trainer.classifier.model.evaluate(train_images, train_labels, verbose=0)
print(f"Training accuracy after adversarial training: {train_acc*100:.2f}%")

test_loss, test_acc = trainer.classifier.model.evaluate(test_images, test_labels, verbose=0)
print(f"Test accuracy (clean data) after adversarial training: {test_acc*100:.2f}%")
trainer.classifier.model.save('mnist_trained_model.h5')
