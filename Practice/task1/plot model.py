from tensorflow.keras.utils import plot_model
import tensorflow as tf
import numpy as np
import keras
model = tf.keras.models.load_model("mnist_model.h5")
plot_model(model, to_file='cnn_architecture.png', show_shapes=True, show_layer_names=True)
