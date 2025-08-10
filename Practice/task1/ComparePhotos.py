from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Load adversarial data
x_test_adv = np.load("x_test_adv.npy")
y_test = np.load("test_labels.npy")
x_test_adv = x_test_adv.reshape((test_images.shape[0], 28, 28, 1))

class_names = [str(i) for i in range(10)]

# --- Evaluate accuracy ---
loss_clean, accuracy_clean = model.evaluate(test_images, y_test, verbose=0)
loss_adv, accuracy_adv = model.evaluate(x_test_adv, y_test, verbose=0)

print(f"Accuracy on clean test examples: {accuracy_clean:.4f}")
print(f"Accuracy on adversarial test examples: {accuracy_adv:.4f}")
print(f"Accuracy drop due to adversarial attack: {accuracy_clean - accuracy_adv:.4f}")

# --- Predictions ---
pred_clean = np.argmax(model.predict(test_images), axis=1)
pred_adv = np.argmax(model.predict(x_test_adv), axis=1)

# --- Comparison table ---
correct_clean = np.sum(pred_clean == y_test)
correct_adv = np.sum(pred_adv == y_test)
print(f"Correct predictions (clean): {correct_clean}")
print(f"Correct predictions (adv): {correct_adv}")
print(f"Wrong predictions (clean): {len(y_test) - correct_clean}")
print(f"Wrong predictions (adv): {len(y_test) - correct_adv}")

# --- Visual comparison ---
plt.figure(figsize=(15, 6))
for i in range(10):
    # Clean image
    plt.subplot(2, 10, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap="gray")
    plt.title(f"Clean:{pred_clean[i]}\nTrue:{y_test[i]}", color=("blue" if pred_clean[i] == y_test[i] else "red"))
    plt.axis("off")

    # Adversarial image
    plt.subplot(2, 10, i + 11)
    plt.imshow(x_test_adv[i].reshape(28, 28), cmap="gray")
    plt.title(f"Adverserial:{pred_adv[i]}\nTrue:{y_test[i]}", color=("blue" if pred_adv[i] == y_test[i] else "red"))
    plt.axis("off")

plt.suptitle("Clean images vs  Adversarial images", fontsize=14)
plt.tight_layout()
plt.show()

# --- Optional: Show where predictions change ---
changed_idx = np.where(pred_clean != pred_adv)[0]
print(f"Number of samples where prediction changed: {len(changed_idx)}")
