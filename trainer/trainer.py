import tensorflow as tf
from data_loader.data_loader import DataLoader
from utils.utils import configInfo, visualization
from model.livenessnet import LivenessNet

class Trainer:

    def __init__(self, dataloader, model, config, batch_size=8, epochs=2):

        self.dataloader = dataloader
        self.model = model
        self.config = configInfo(config)
        self.batch_size = batch_size
        self.epochs = epochs

        self.train_generator, self.validation_generator = self.dataloader.data_generator()

    def train(self):

        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print(self.model.summary())
        history = self.model.fit(self.train_generator, epochs=self.epochs,
                                 validation_data=self.validation_generator, verbose=1, steps_per_epoch=len(self.train_generator) / self.epochs)
        #

        self.model.save(self.config["saved_model"])

        return history

if __name__ == "__main__":
    dataloader = DataLoader(config="../config.json")
    model = LivenessNet().build()
    trainer = Trainer(dataloader, model, config="../config.json")
    history = trainer.train()
    visualization(history)
