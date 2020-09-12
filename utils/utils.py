import json
import matplotlib.pyplot as plt

def configInfo(file):
    with open(file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def visualization(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    # plt.figure()

    plt.show()