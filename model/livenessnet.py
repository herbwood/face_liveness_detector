import tensorflow as tf

class LivenessNet:
  def __init__(self, width=32, height=32, depth=3, classes=1):
    self.width = width
    self.height = height
    self.depth = depth
    self.classes = classes

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

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Dense(self.classes, activation="sigmoid"))

    return model

if __name__ == "__main__":
    model = LivenessNet().build()
    print(model.summary())