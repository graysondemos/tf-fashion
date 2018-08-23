import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv("fashion-mnist_train.csv", sep = ",")
test = pd.read_csv("fashion-mnist_test.csv", sep = ",")

train = train.values
test = test.values

train_labels = train[:, 0]
test_labels = test[:, 0]
train = train[:, 1:]
test = test[:, 1:]

train = train.reshape(60000, 28, 28)
test = test.reshape(10000, 28, 28)
train = train / 255.0
test = test / 255.0

model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28, 28)),
	keras.layers.Dense(128, activation = tf.nn.relu),
	keras.layers.Dense(10, activation = tf.nn.softmax)])

model.compile(optimizer = tf.train.AdamOptimizer(), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(train, train_labels, epochs = 5)

img = tf.image.decode_jpeg(tf.read_file("test5.jpg"), channels = 1)
sess = tf.InteractiveSession()
img = sess.run(img)
sess.close()
img = img.reshape(1, 28, 28)
img = img / 255.0

labels = ["T-Shirt", "Pants", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

prediction = model.predict(img)
print(labels[np.argmax(prediction[0])])