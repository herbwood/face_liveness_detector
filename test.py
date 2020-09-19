import tensorflow as tf
from utils.utils import configInfo
from data_loader.data_loader import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def test(dataloader, config="config.json"):
    config = configInfo(config)
    model = tf.keras.models.load_model(config["best_saved_model"])

    _, validation_generator = dataloader.data_generator()
    batch_size = dataloader.batch_size
    target_names = list(dataloader.labels.keys())

    y_pred = model.predict(validation_generator, validation_generator.samples // batch_size + 1)
    y_pred = np.argmax(y_pred, axis=1)

    print('Classification Report')
    print(classification_report(validation_generator.classes, y_pred))

if __name__ == "__main__":
    dataloader = DataLoader(config="config.json")
    test(dataloader)