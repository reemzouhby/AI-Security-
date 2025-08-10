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

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # by default 85% training and  15% for testing 60000 for training and 10000 for teting
print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']
FULLY_CONNECT_NUM = 128
batch_size=128

NUM_CLASSES = len(class_names)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


history = model.fit(train_images, train_labels,
                    epochs=10,
                    batch_size=128,
                    validation_data=(test_images, test_labels),
                    verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc*100:.2f}%")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

metrics_df = pd.DataFrame(history.history)
metrics_df[["loss", "val_loss"]].plot()
plt.show()
# Save the model
model.save('mnist_model.h5')
# Add a channel dimension for grayscale images
test_images_reshaped = test_images.reshape((test_images.shape[0], 28, 28, 1))

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
    plt.imshow(test_images[i], cmap=plt.cm.gray)
    true_label = class_names[test_labels[i]]
    predicted_label = class_names[predicted_labels[i]]
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
plt.tight_layout()
plt.show()
print(f"Max prediction accuracy: {np.max(predictions)}")
print(f"Number of correct predictions: {np.sum(np.argmax(predictions, axis=1) == test_labels)}")
print(f"Number of  wrong prediction : {len(test_labels) - np.sum(np.argmax(predictions, axis=1) == test_labels)}")
print(f"Total test samples: {len(test_labels)}")