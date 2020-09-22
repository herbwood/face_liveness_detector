import json
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def configInfo(file):
    with open(file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def visualization(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # plt.plot(epochs, acc, 'r', label='Training accuracy')
    # plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend(loc=0)

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    # plt.figure()

    plt.show()


def convert_to_tflite(h5_model, export_path, save_dir, filename):
    model = tf.keras.models.load_model(h5_model, compile=False)
    model.save(export_path, save_format="tf")

    converter = tf.lite.TFLiteConverter.from_saved_model(export_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    open(os.path.join(save_dir, filename), "wb").write(tflite_model)
