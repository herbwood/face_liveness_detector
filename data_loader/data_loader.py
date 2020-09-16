import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.utils import configInfo

class DataLoader:

    def __init__(self, config, batch_size=32, target_size=(64, 64)):
        self.config = configInfo(config)
        self.batch_size = batch_size
        self.target_size = target_size
        train_dir = self.config["train_dir"]
        validation_dir = self.config["validation_dir"]

        self.train_datagen = ImageDataGenerator()
        # rescale = 1 / 255,
        # rotation_range = 20,
        # width_shift_range = 0.2,
        # height_shift_range = 0.2,
        # shear_range = 0.15,
        # zoom_range = 0.15,
        # horizontal_flip = True,
        # fill_mode = 'nearest'

        self.validation_datagen = ImageDataGenerator()
        # rescale = 1 / 255

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
    dl = DataLoader(config="../config.json")
    train_generator, validation_generator = dl.data_generator()
    print(dl.labels)