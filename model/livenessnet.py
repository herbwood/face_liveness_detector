import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from utils.utils import configInfo

class LivenessNet:
  def __init__(self, config, width=64, height=64, depth=3, classes=2):
    self.config = configInfo(config)
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

    model.add(tf.keras.layers.Dense(self.classes, activation="softmax"))

    return model

    # pre_trained_model = InceptionV3(input_shape = (self.width, self.height, self.depth),
    #                             include_top = False,
    #                             weights = None)
    # local_weights_file = self.config["local_weights_file"]
    # pre_trained_model.load_weights(local_weights_file)
    #
    # for layer in pre_trained_model.layers:
    #   layer.trainable = False
    #
    # last_layer = pre_trained_model.get_layer('mixed7')
    # print('last layer output shape :', last_layer.output_shape)
    # last_output = last_layer.output
    #
    # # Flatten the output layer to 1 dimension
    # x = tf.keras.layers.Flatten()(last_output)
    #
    # # Add a fully connected layer with 1,024 hidden units and ReLU activation
    # x = tf.keras.layers.Dense(1024, activation='relu')(x)
    #
    # # Add a dropout rate of 0.2
    # x = tf.keras.layers.Dropout(0.2)(x)
    #
    # # Add a final sigmoid layer for classification
    # x = tf.keras.layers.Dense(self.classes, activation='sigmoid')(x)
    #
    # model = tf.keras.Model(pre_trained_model.input, x)
    #
    # return model


# if __name__ == "__main__":
#     model = LivenessNet().build()
#     print(model.summary())