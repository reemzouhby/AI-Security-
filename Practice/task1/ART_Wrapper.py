

import tensorflow as tf
from art.estimators.classification import KerasClassifier

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

# Create ART KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

print("ART KerasClassifier created successfully.")
