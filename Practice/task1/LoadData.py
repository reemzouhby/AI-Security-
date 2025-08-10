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
plt.figure(figsize=(10, 10))

#print first 25 images from training data
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],'gray')
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#print second 25 from training data
plt.figure(figsize=(10, 10))
j=0
for i in range(25,50,1):
    j=j+1
    plt.subplot(5, 5, j)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],'gray')
    plt.xlabel(class_names[train_labels[i]])

plt.show()