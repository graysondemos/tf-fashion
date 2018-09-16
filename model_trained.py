import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

model = keras.models.load_model("model.h5")
model.compile(optimizer = tf.train.AdamOptimizer(), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

test_path = input("Test Path: ")

img = tf.image.decode_jpeg(tf.read_file(test_path), channels = 1)
sess = tf.InteractiveSession()
img = sess.run(img)
sess.close()
img = img.reshape(1, 28, 28)
img = img / 255.0

labels = ["T-Shirt", "Pants", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

predictions = model.predict(img)
print(labels[np.argmax(predictions[0])])