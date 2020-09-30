import tensorflow as tf
from utils.utils import configInfo
from data_loader.data_loader import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def test(dataloader, config="config/config.json"):
    config = configInfo(config)
    model = tf.keras.models.load_model(config["trial_saved_model"])

    _, validation_generator = dataloader.data_generator()
    batch_size = dataloader.batch_size
    target_names = list(dataloader.labels.keys())

    y_pred = model.predict(validation_generator, validation_generator.samples // batch_size + 1)
    y_pred = np.argmax(y_pred, axis=1)

    print('Classification Report')
    print(classification_report(validation_generator.classes, y_pred))

def logTest(config):
    with open(config) as f:
        logs = f.readlines()

    total = len(logs)
    rkc, ruc, fkc, fuc = 0, 0, 0, 0

    for log in logs:
        log = log.strip()
        name, label, prob, width, height, channel = log.split()[2:]

        if name != "Unknown" and label == "real":
            rkc += 1
        elif name == "Unknown" and label == "real":
            ruc += 1
        elif name != "Unknown" and label == "fake":
            fkc += 1
        else:
            fuc += 1


    print("{:^12}|{:^12}|{:^12}|{:^12}|".format("", "real", "fake", "total"))
    print("{:^12}|{:^12}|{:^12}|{:^12}|".format("known", f"{rkc}({round(rkc / total, 2)})",
                                                f"{fkc}({round(fkc/total, 2)})", f"{rkc + fkc}({round((rkc + fkc)/total, 2)})"))
    print("{:^12}|{:^12}|{:^12}|{:^12}|".format("unknown", f"{ruc}({round(ruc / total, 2)})",
                                                f"{fuc}({round(fuc/total, 2)})", f"{ruc + fuc}({round((ruc + fuc)/total, 2)})"))
    print("{:^12}|{:^12}|{:^12}|{:^12}|".format("total", f"{rkc + ruc}({round((rkc + ruc)/total, 2)})",
                                                f"{fkc + fuc}({round((fkc + fuc)/total, 2)})", f"{total}({round((ruc + fuc + rkc + fkc)/total, 2)})"))
    
    if round(rkc / total, 2) >= 0.65:
        return "accept"
    return "denied"


if __name__ == "__main__":
    dataloader = DataLoader(config="config/config.json")
    # test(dataloader)
    logTest("config/logs_2020_09_29_11_13_09.log")