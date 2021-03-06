# import tensorflow as tf
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from utils.utils import configInfo
#
# class TrialNet:
#     def __init__(self, config, width=128, height=128, depth=3, classes=2):
#         self.config = configInfo(config)
#         self.width = width
#         self.height = height
#         self.depth = depth
#         self.classes = classes
#
#     def build(self):
#         pre_trained_model = InceptionV3(input_shape=(self.width, self.height, self.depth),
#                                         include_top=False,
#                                         weights="imagenet")
#         # local_weights_file = self.config["local_weights_file"]
#         # pre_trained_model.load_weights(local_weights_file)
#
#         for layer in pre_trained_model.layers:
#             layer.trainable = False
#
#         last_layer = pre_trained_model.get_layer('mixed7')
#         print('last layer output shape :', last_layer.output_shape)
#         last_output = last_layer.output
#
#         # Flatten the output layer to 1 dimension
#         x = tf.keras.layers.Flatten()(last_output)
#
#         # Add a fully connected layer with 1,024 hidden units and ReLU activation
#         x = tf.keras.layers.Dense(1024, activation='relu')(x)
#
#         # Add a dropout rate of 0.2
#         x = tf.keras.layers.Dropout(0.2)(x)
#
#         # Add a final sigmoid layer for classification
#         x = tf.keras.layers.Dense(self.classes, activation='sigmoid')(x)
#
#         model = tf.keras.Model(pre_trained_model.input, x)
#
#         return model

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from utils.utils import configInfo

class TrialNet:
  def __init__(self, config):
    self.config = configInfo(config)
    self.hyperparameters = self.config["hyperparameters"]
    self.width, self.height, self.depth = self.hyperparameters["size"]
    self.classes = self.config["le"]["num_classes"]

  def build(self):
    pre_trained_model = DenseNet121(input_shape=(self.width, self.height, self.depth),
                                          include_top=False,
                                          weights="imagenet")
          # local_weights_file = self.config["local_weights_file"]
          # pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
          layer.trainable = False

    last_layer = pre_trained_model.get_layer('conv4_block17_concat')
    print('last layer output shape :', last_layer.output_shape)
    last_output = last_layer.output

          # Flatten the output layer to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)

          # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(256, activation='relu')(x)

          # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2)(x)

          # Add a final sigmoid layer for classification
    x = tf.keras.layers.Dense(self.classes, activation='softmax')(x)

    model = tf.keras.Model(pre_trained_model.input, x)

    return model
    # return pre_trained_model


if __name__ == "__main__":
  trainer = TrialNet("../config.json").build()
  print(trainer.summary())