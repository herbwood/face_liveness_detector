import tensorflow as tf
from data_loader.data_loader import DataLoader
from utils.utils import configInfo, visualization
from model.livenessnet import LivenessNet
from model.trialnet import TrialNet

class Trainer:

    def __init__(self, dataloader, model, config):

        self.dataloader = dataloader
        self.model = model
        self.config = configInfo(config)
        self.hyperparameters = self.config["hyperparameters"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.epochs = self.hyperparameters["epochs"]
        self.INIT_LR = self.hyperparameters["learning_rate"]
        self.optimizer = tf.keras.optimizers.Adam(lr=self.INIT_LR, decay=self.INIT_LR/self.epochs)

        self.train_generator, self.validation_generator = self.dataloader.data_generator()

        self.step_size_train = self.train_generator.n // self.train_generator.batch_size
        self.step_size_validation = self.validation_generator.samples // self.validation_generator.batch_size

    def train(self):

        self.model.compile(optimizer=self.optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        print(self.model.summary())

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)

        history = self.model.fit(self.train_generator, epochs=self.epochs,
                                 validation_data=self.validation_generator, verbose=1,
                                 steps_per_epoch=self.step_size_train,
                                 validation_steps=self.step_size_validation,
                                 callbacks=[early_stopping])
        #, steps_per_epoch=len(self.train_generator) / self.epochs

        self.model.save(self.config["trial_saved_model"])

        return history

if __name__ == "__main__":
    dl = DataLoader("../config.json")
    model = LivenessNet("../config.json").build()
    trainer = Trainer(dl, model, config="../config.json")
    history = trainer.train()
