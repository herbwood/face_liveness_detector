import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.utils import configInfo

class DataLoader:

    def __init__(self, config):
        self.config = configInfo(config)
        self.hyperparameters = self.config["hyperparameters"]
        self.width, self.height, self.channel = self.hyperparameters["size"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.target_size = (self.width, self.height)
        train_dir = self.config["train_dir"]
        validation_dir = self.config["validation_dir"]
        # train_dir = "../dataset/face_liveness_train"
        # validation_dir = "../dataset/face_liveness_validation"

        self.train_datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])

        self.validation_datagen = ImageDataGenerator()

        self.train_generator = self.train_datagen.flow_from_directory(train_dir,
                                                            batch_size=self.batch_size,
                                                            target_size=self.target_size,
                                                            color_mode = "rgb",
                                                            class_mode='binary',
                                                            shuffle=True)

        self.validation_generator = self.validation_datagen.flow_from_directory(validation_dir,
                                                                      batch_size=self.batch_size,
                                                                      target_size=self.target_size,
                                                                      color_mode="rgb",
                                                                      class_mode='binary',
                                                                      shuffle=False)

        self.labels = self.train_generator.class_indices


    def data_generator(self):
        return self.train_generator, self.validation_generator

if __name__ == "__main__":
    dl = DataLoader("../config.json")
    print(dl.data_generator())
