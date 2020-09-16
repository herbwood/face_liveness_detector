from data_loader.data_loader import DataLoader
from utils.utils import configInfo, visualization
from model.livenessnet import LivenessNet
from trainer.trainer import Trainer

def main():
    dataloader = DataLoader(config="config.json")
    model = LivenessNet(config="config.json").build()
    trainer = Trainer(dataloader, model, config="config.json")
    history = trainer.train()
    visualization(history)

if __name__ == "__main__":
    main()