import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# print("Version : ", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# predictions = model(x_train[0:1]).numpy()
# tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_fn(y_train[0:1], predictions).numpy()


model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=["accuracy"]
)

history = model.fit(x_train, y_train, epochs=5)
accuracy_list = history.history["accuracy"]
loss_list = history.history["loss"]
epochs = range(1, len(accuracy_list) + 1)

model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.models.Sequential([
    model,
    tf.keras.layers.Softmax()
])

pred = probability_model(x_test[:1])
print("Predictions : ", pred)
print("Total : ", np.sum(pred))


# plt.plot(accuracy_list, marker='o')
# plt.title("Accuracy vs Epochs")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.grid(True)
# plt.show()

# Accuracy plot
plt.plot(epochs, accuracy_list, marker='o')
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Loss plot
plt.plot(epochs, loss_list, marker='o', color='r')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
