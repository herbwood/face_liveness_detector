from data_loader.data_loader import DataLoader
from utils.utils import configInfo, visualization
from model.livenessnet import LivenessNet
from trainer.trainer import Trainer
from model.trialnet import TrialNet
import argparse
from utils.utils import convert_to_tflite

# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", type=str, required=True, help="name of the model")
# ap.add_argument("-c", "--convert", type=str, required=True, help="convert trained model to tflite")
# args = vars(ap.parse_args())

def train():

    dataloader = DataLoader(config="config/config.json")
    # if args["model"] == "trialnet":
    model = TrialNet(config="config/config.json").build()
    # else:
    #     model = LivenessNet(config="config.json").build()
    trainer = Trainer(dataloader, model, config="config/config.json")
    history = trainer.train()
    # if args["convert"] == "yes":
    #     convert_to_tflite()
    visualization(history)

if __name__ == "__main__":
    # python train.py --model trialnet --convert True
    train()
