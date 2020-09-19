import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from utils.utils import configInfo

class LivenessNet:
  def __init__(self, config):
    self.config = configInfo(config)
    self.hyperparameters = self.config["hyperparameters"]
    self.width, self.height, self.depth = self.hyperparameters["size"]
    self.classes = self.config["le"]["num_classes"]

  def build(self):
    input_shape = (self.height, self.width, self.depth)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Dense(self.classes, activation="softmax"))

    return model

if __name__ == "__main__":
  model = LivenessNet("../config.json").build()
  print(model.summary())