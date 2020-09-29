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
    real_known_cnt, real_unknown_cnt,  fake_known_cnt, fake_unknown_cnt = 0, 0, 0, 0

    for log in logs:
        log = log.strip()
        name, label, prob, width, height, channel = log.split()[2:]

        if name != "Unknown" and label == "real":
            real_known_cnt += 1
        elif name == "Unknown" and label == "real":
            real_unknown_cnt += 1
        elif name != "Unknown" and label == "fake":
            fake_known_cnt += 1
        else:
            fake_unknown_cnt += 1



    print(f"          |    real   |     fake    |   total     |")
    print(f"| known   | {real_known_cnt}({round(real_known_cnt / total, 2)}) |   {fake_known_cnt}({round(fake_known_cnt/total, 2)})  | {real_known_cnt + fake_known_cnt}({round((real_known_cnt + fake_known_cnt)/total, 2)})   |")
    print(f"| unknown | {real_unknown_cnt}({round(real_unknown_cnt/total, 2)})   |   {fake_unknown_cnt}({round(fake_unknown_cnt/total, 2)})   |  {real_unknown_cnt + fake_unknown_cnt}({round((real_unknown_cnt + fake_unknown_cnt)/total, 2)})   |")
    print(f"| total   | {real_known_cnt + real_unknown_cnt}({round((real_known_cnt + real_unknown_cnt)/total, 2)})  |    {fake_known_cnt + fake_unknown_cnt}({round((fake_known_cnt + fake_unknown_cnt)/total, 2)})  | {total}(1)      |")

    if round(real_known_cnt / total, 2) >= 0.65:
        return "accept"
    return "denied"


if __name__ == "__main__":
    dataloader = DataLoader(config="config/config.json")
    # test(dataloader)
    logTest("config/logs_2020_09_29_11_13_09.log")