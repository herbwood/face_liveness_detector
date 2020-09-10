from data_loader.data_generator import DataLoader
from model.livenessnet import LivenessNet
from trainer.trainer import Trainer
from utils.utils import configInfo

def main():
    dataloader = DataLoader(config="config.json")
    model = LivenessNet.build(width=32, height=32, depth=3, classes=2)
    trainer = Trainer(dataloader, config="config.json")
    trainer.train()

if __name__ == "__main__":
    main()